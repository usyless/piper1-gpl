#include "piper.hpp"

#include <fstream>
#include "json.hpp"
#include "onnxruntime_cxx_api.h"
#include "uni_algo.h"
#include <thread>
#include <string_view>

#include <espeak-ng/speak_lib.h>

static Ort::Env ort_env{ORT_LOGGING_LEVEL_WARNING, "piper"};

namespace piper {

inline std::optional<Phoneme> get_codepoint(const std::string& s) {
    auto view = una::views::utf8(s);
    auto it = view.begin();

    if (it != view.end()) {
        return *it;
    }

    return std::nullopt;
}

using json = nlohmann::json;

std::unique_ptr<Synthesizer> Synthesizer::create(const std::string& model_path, const std::string& config_path, const std::string& espeak_data_path, std::optional<Ort::SessionOptions> options) {
    try {
    
    if (model_path.empty()) {
        return nullptr;
    }

    std::ifstream config_stream(config_path.empty() ? model_path + ".json" : config_path);
    auto config = json::parse(config_stream);

    if (espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, espeak_data_path.c_str(), 0) < 0) {
        return nullptr;
    }

    auto synth_ptr = std::make_unique<Synthesizer>();
    auto& synth = *synth_ptr;

    // Load config options
    if (config.contains("espeak")) {
        const auto &espeak_obj = config.at("espeak");
        if (espeak_obj.contains("voice")) {
            espeak_obj.at("voice").get_to(synth.espeak_voice);
        }
    }

    if (config.contains("audio")) {
        const auto &audio_obj = config.at("audio");
        if (audio_obj.contains("sample_rate")) {
            // Sample rate of generated audio in hertz
            audio_obj.at("sample_rate").get_to(synth.sample_rate);
        }
    }

    // phoneme to [id] map
    // Maps phonemes to one or more phoneme ids (required).
    if (config.contains("phoneme_id_map")) {
        const auto& phoneme_id_map_value = config.at("phoneme_id_map");
        for (const auto& from_phoneme_item : phoneme_id_map_value.items()) {
            const auto from_codepoint = get_codepoint(from_phoneme_item.key());
            if (!from_codepoint) continue;

            auto& from_codepoint_map = synth.phoneme_id_map[*from_codepoint];

            for (const auto& to_id_value : from_phoneme_item.value()) {
                from_codepoint_map.emplace_back(to_id_value.get<PhonemeId>());
            }
        }
    }

    synth.num_speakers = config.at("num_speakers").get<SpeakerId>();

    if (config.contains("inference")) {
        // Overrides default inference settings
        const auto& inference_value = config.at("inference");
        if (inference_value.contains("noise_scale")) {
            inference_value.at("noise_scale").get_to(synth.default_options.noise_scale);
        }

        if (inference_value.contains("length_scale")) {
            inference_value.at("length_scale").get_to(synth.default_options.length_scale);
        }

        if (inference_value.contains("noise_w")) {
            inference_value.at("noise_w").get_to(synth.default_options.noise_w_scale);
        }
    }

    synth.set_default_options();

    if (options) {
        synth.session_options = std::move(*options);
    } else {
        synth.session_options.DisableCpuMemArena();
        synth.session_options.DisableMemPattern();
        synth.session_options.DisableProfiling();
        synth.session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        synth.session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        synth.session_options.SetInterOpNumThreads(1);
        synth.session_options.SetIntraOpNumThreads((int)std::thread::hardware_concurrency());
    }

    synth.session = std::make_unique<Ort::Session>(ort_env, model_path.c_str(), synth.session_options);

    return synth_ptr;

    } catch (...) {
        return nullptr;
    }
}

Synthesizer::~Synthesizer() {
    espeak_Terminate();
}

bool Synthesizer::start(const std::string& text, phoneme_id_queue_t& phoneme_id_queue) {
    if (espeak_SetVoiceByName(espeak_voice.c_str()) != EE_OK) {
        return false;
    }

    // phonemize
    std::vector<std::string> sentence_phonemes{""};
    std::size_t current_idx = 0;
    const void *text_ptr = text.c_str();
    while (text_ptr != nullptr) {
        int terminator = 0;
        std::string_view terminator_str{""};

        const char* phonemes = espeak_TextToPhonemesWithTerminator(
            &text_ptr, espeakCHARS_AUTO, espeakPHONEMES_IPA, &terminator);

        if (phonemes) {
            sentence_phonemes[current_idx] += phonemes;
        }

        // Categorize terminator
        terminator &= 0x000FFFFF;

        if (terminator == CLAUSE_PERIOD) {
            terminator_str = ".";
        } else if (terminator == CLAUSE_QUESTION) {
            terminator_str = "?";
        } else if (terminator == CLAUSE_EXCLAMATION) {
            terminator_str = "!";
        } else if (terminator == CLAUSE_COMMA) {
            terminator_str = ", ";
        } else if (terminator == CLAUSE_COLON) {
            terminator_str = ": ";
        } else if (terminator == CLAUSE_SEMICOLON) {
            terminator_str = "; ";
        }

        sentence_phonemes[current_idx] += terminator_str;

        if ((terminator & CLAUSE_TYPE_SENTENCE) == CLAUSE_TYPE_SENTENCE) {
            sentence_phonemes.emplace_back();
            current_idx = sentence_phonemes.size() - 1;
        }
    }

    // phonemes to ids
    std::vector<Phoneme> sentence_codepoints;
    std::vector<PhonemeId> sentence_ids;
    phoneme_id_queue.reserve(sentence_phonemes.size());
    for (auto &phonemes_str : sentence_phonemes) {
        if (phonemes_str.empty()) {
            continue;
        }

        sentence_codepoints.push_back(PHONEME_BOS);
        sentence_ids.push_back(ID_BOS);

        sentence_codepoints.push_back(PHONEME_BOS);
        sentence_ids.push_back(ID_PAD);

        sentence_codepoints.push_back(PHONEME_SEPARATOR);

        auto phonemes_norm = una::norm::to_nfd_utf8(phonemes_str);
        auto phonemes_range = una::ranges::utf8_view{phonemes_norm};
        auto phonemes_iter = phonemes_range.begin();
        auto phonemes_end = phonemes_range.end();

        // Filter out (lang) switch (flags).
        // These surround words from languages other than the current voice.
        bool in_lang_flag = false;
        while (phonemes_iter != phonemes_end) {
            auto phoneme = *phonemes_iter;

            if (in_lang_flag) {
                if (phoneme == U')') {
                    // End of (lang) switch
                    in_lang_flag = false;
                }
            } else if (phoneme == U'(') {
                // Start of (lang) switch
                in_lang_flag = true;
            } else {
                // Look up ids
                auto ids_for_phoneme = this->phoneme_id_map.find(phoneme);
                if (ids_for_phoneme != this->phoneme_id_map.end()) {
                    for (auto id : ids_for_phoneme->second) {
                        sentence_codepoints.push_back(phoneme);
                        sentence_ids.push_back(id);

                        sentence_codepoints.push_back(phoneme);
                        sentence_ids.push_back(ID_PAD);

                        sentence_codepoints.push_back(PHONEME_SEPARATOR);
                    }
                }
            }

            phonemes_iter++;
        }

        sentence_codepoints.push_back(PHONEME_EOS);
        sentence_ids.push_back(ID_EOS);
        sentence_codepoints.push_back(PHONEME_SEPARATOR);

        phoneme_id_queue.emplace_back(sentence_codepoints, std::move(sentence_ids));
    }

    return true;
}

}