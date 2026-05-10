#include "piper.hpp"

#include <array>
#include <fstream>
#include <limits>
#include "json.hpp"
#include "onnxruntime_cxx_api.h"
#include <thread>

#include <espeak-ng/speak_lib.h>

static Ort::Env ort_env{ORT_LOGGING_LEVEL_WARNING, "piper"};

namespace piper {

using json = nlohmann::json;

std::unique_ptr<Synthesizer> Synthesizer::create(const std::string& model_path, const std::string& config_path, const std::string& espeak_data_path, std::optional<Ort::SessionOptions> options) {
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
            auto from_codepoint = get_codepoint(from_phoneme_item.key());
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
            inference_value.at("noise_scale").get_to(synth.options.noise_scale);
        }

        if (inference_value.contains("length_scale")) {
            inference_value.at("length_scale").get_to(synth.options.length_scale);
        }

        if (inference_value.contains("noise_w")) {
            inference_value.at("noise_w").get_to(synth.options.noise_w_scale);
        }
    }

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
}

Synthesizer::~Synthesizer() {
    espeak_Terminate();
}

int Synthesizer::start(const std::string& text) {
    if (espeak_SetVoiceByName(espeak_voice.c_str()) != EE_OK) {
        return PIPER_ERR_GENERIC;
    }

    // Clear state
    while (!this->phoneme_id_queue.empty()) {
        this->phoneme_id_queue.pop();
    }
    this->chunk_samples.clear();

    // phonemize
    std::vector<std::string> sentence_phonemes{""};
    std::size_t current_idx = 0;
    const void *text_ptr = text.c_str();
    while (text_ptr != nullptr) {
        int terminator = 0;
        std::string terminator_str = "";

        const char *phonemes = espeak_TextToPhonemesWithTerminator(
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
            sentence_phonemes.push_back("");
            current_idx = sentence_phonemes.size() - 1;
        }
    }

    // phonemes to ids
    std::vector<Phoneme> sentence_codepoints;
    std::vector<PhonemeId> sentence_ids;
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

        this->phoneme_id_queue.emplace(sentence_codepoints, std::move(sentence_ids));
        sentence_ids.clear();
    }

    return PIPER_OK;
}

int Synthesizer::next(AudioChunk& chunk) {
    // Clear data from previous call
    this->chunk_samples.clear();
    this->chunk_phonemes.clear();
    this->chunk_phoneme_ids.clear();
    this->chunk_alignments.clear();

    chunk.sample_rate = this->sample_rate;
    chunk.samples = nullptr;
    chunk.num_samples = 0;
    chunk.is_last = false;
    chunk.phoneme_ids = nullptr;
    chunk.num_phoneme_ids = 0;
    chunk.alignments = nullptr;
    chunk.num_alignments = 0;

    if (this->phoneme_id_queue.empty()) {
        // Empty final chunk
        chunk.is_last = true;
        return PIPER_DONE;
    }

    // Process next list of phoneme ids
    auto [next_phonemes, next_ids] = std::move(this->phoneme_id_queue.front());
    this->phoneme_id_queue.pop();

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Allocate
    std::vector<int64_t> phoneme_id_lengths{(int64_t)next_ids.size()};
    std::vector<float> scales{this->options.noise_scale, this->options.length_scale,
                              this->options.noise_w_scale};

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve((this->num_speakers > 1) ? 5 : 4);
    std::vector<int64_t> phoneme_ids_shape{1, (int64_t)next_ids.size()};
    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, next_ids.data(), next_ids.size(), phoneme_ids_shape.data(),
        phoneme_ids_shape.size()));

    std::vector<int64_t> phoneme_id_lengths_shape{
        (int64_t)phoneme_id_lengths.size()};
    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, phoneme_id_lengths.data(), phoneme_id_lengths.size(),
        phoneme_id_lengths_shape.data(), phoneme_id_lengths_shape.size()));

    std::vector<int64_t> scales_shape{(int64_t)scales.size()};
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memoryInfo, scales.data(), scales.size(), scales_shape.data(),
        scales_shape.size()));

    // Add speaker id.
    // NOTE: These must be kept outside the "if" below to avoid being
    // deallocated.
    std::vector<int64_t> speaker_id{(int64_t)this->options.speaker_id};
    std::vector<int64_t> speaker_id_shape{(int64_t)speaker_id.size()};

    if (this->num_speakers > 1) {
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memoryInfo, speaker_id.data(), speaker_id.size(),
            speaker_id_shape.data(), speaker_id_shape.size()));
    }

    // From export_onnx.py
    std::array<const char *, 4> input_names = {"input", "input_lengths",
                                               "scales", "sid"};

    // Get all output names
    const auto output_names_strs = this->session->GetOutputNames();
    std::vector<const char *> output_names;
    output_names.reserve(output_names_strs.size());
    for (const auto &name : output_names_strs) {
        output_names.push_back(name.c_str());
    }

    // Infer
    auto output_tensors = this->session->Run(
        Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
        input_tensors.size(), output_names.data(), output_names.size());

    if ((output_tensors.size() < 1) || (!output_tensors.front().IsTensor())) {
        return PIPER_ERR_GENERIC;
    }

    auto audio_shape =
        output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
    chunk.num_samples = audio_shape[audio_shape.size() - 1];

    const float *audio_tensor_data = output_tensors.front().GetTensorData<float>();
    this->chunk_samples.resize(chunk.num_samples);
    std::copy(audio_tensor_data, audio_tensor_data + chunk.num_samples,
              this->chunk_samples.begin());
    chunk.samples = this->chunk_samples.data();

    chunk.is_last = this->phoneme_id_queue.empty();

    // Copy phonemes
    this->chunk_phonemes = std::move(next_phonemes);
    chunk.phonemes = this->chunk_phonemes.data();
    chunk.num_phonemes = this->chunk_phonemes.size();

    // Copy phoneme ids
    for (auto phoneme_id : next_ids) {
        if (phoneme_id < std::numeric_limits<int>::min() ||
            phoneme_id > std::numeric_limits<int>::max()) {
            continue;
        }
        this->chunk_phoneme_ids.push_back(static_cast<int>(phoneme_id));
    }

    chunk.phoneme_ids = this->chunk_phoneme_ids.data();
    chunk.num_phoneme_ids = this->chunk_phoneme_ids.size();

    // Check for alignments
    if (output_tensors.size() > 1) {
        auto alignments_shape =
            output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

        chunk.num_alignments = alignments_shape[alignments_shape.size() - 1];
        const float *alignments_tensor_data =
            output_tensors[1].GetTensorData<float>();

        this->chunk_alignments.resize(chunk.num_alignments);
        for (std::size_t i = 0; i < chunk.num_alignments; i++) {
            this->chunk_alignments[i] =
                (int)(alignments_tensor_data[i] * this->hop_length);
        }

        chunk.alignments = this->chunk_alignments.data();
    }

    // Clean up
    for (auto& tensor : output_tensors) {
        Ort::detail::OrtRelease(tensor.release());
    }

    for (auto& tensor : input_tensors) {
        Ort::detail::OrtRelease(tensor.release());
    }

    return PIPER_OK;
}

}