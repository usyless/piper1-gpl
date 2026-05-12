#pragma once

#include <functional>
#include <span>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <map>
#include <memory>
#include <optional>
#include <array>
#include <stdint.h>
#include <string>
#include <vector>

#ifdef LIBPIPER_FULL_AUDIOCHUNK
#include <limits>
#endif

#include <onnxruntime_cxx_api.h>

// espeak
#define CLAUSE_INTONATION_FULL_STOP 0x00000000
#define CLAUSE_INTONATION_COMMA 0x00001000
#define CLAUSE_INTONATION_QUESTION 0x00002000
#define CLAUSE_INTONATION_EXCLAMATION 0x00003000

#define CLAUSE_TYPE_CLAUSE 0x00040000
#define CLAUSE_TYPE_SENTENCE 0x00080000

#define CLAUSE_PERIOD (40 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_COMMA (20 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE)
#define CLAUSE_QUESTION (40 | CLAUSE_INTONATION_QUESTION | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_EXCLAMATION                                                     \
    (45 | CLAUSE_INTONATION_EXCLAMATION | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_COLON (30 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_CLAUSE)
#define CLAUSE_SEMICOLON (30 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE)

namespace piper {
    using Phoneme = char32_t;
    using PhonemeId = int64_t;
    using SpeakerId = int64_t;
    using PhonemeIdMap = std::map<Phoneme, std::vector<PhonemeId>>;

    constexpr inline PhonemeId ID_PAD = 0; // interleaved
    constexpr inline PhonemeId ID_BOS = 1; // beginning of sentence
    constexpr inline PhonemeId ID_EOS = 2; // end of sentence

    constexpr inline Phoneme PHONEME_PAD = U'_';
    constexpr inline Phoneme PHONEME_BOS = U'^';
    constexpr inline Phoneme PHONEME_EOS = U'$';
    constexpr inline Phoneme PHONEME_SEPARATOR = 0;

    constexpr inline float DEFAULT_LENGTH_SCALE = 1.0f;
    constexpr inline float DEFAULT_NOISE_SCALE = 0.667f;
    constexpr inline float DEFAULT_NOISE_W_SCALE = 0.8f;

    constexpr inline int DEFAULT_HOP_LENGTH = 256;

    /**
    * \brief Options for synthesis.
    *
    * \sa \ref piper_default_synthesize_options
    */
    struct SynthesizerOptions {
        /**
        * \brief Id of speaker to use (multi-speaker models only).
        *
        * Id 0 is the first speaker.
        */
        SpeakerId speaker_id;

        /**
        * \brief How fast the text is spoken.
        *
        * A length scale of 0.5 means to speak twice as fast.
        * A length scale of 2.0 means to speak twice as slow.
        * The default is 1.0.
        */
        float length_scale;

        /**
        * \brief Controls how much noise is added during synthesis.
        *
        * The best value depends on the voice.
        * For single speaker models, a value of 0.667 is usually good.
        * For multi-speaker models, a value of 0.333 is usually good.
        */
        float noise_scale;

        /**
        * \brief Controls how much phonemes vary in length during synthesis.
        *
        * The best value depends on the voice.
        * For single speaker models, a value of 0.8 is usually good.
        * For multi-speaker models, a value of 0.333 is usually good.
        */
        float noise_w_scale;
    };

    /**
    * \brief Chunk of synthesized audio samples.
    */
    struct AudioChunk {
        /**
        * \brief Raw samples returned from the voice model.
        */
        std::span<const float> samples{};

        /**
        * \brief Sample rate in Hertz.
        */
        int sample_rate{0};

        /**
        * \brief True if this is the last audio chunk.
        */
        bool is_last{false};

        #ifdef LIBPIPER_FULL_AUDIOCHUNK

        /**
        * \brief Phoneme codepoints that produced this audio chunk, aligned with ids.
        *
        * Phonemes will look like [p1, p1, 0, p2, p2, 0, ...] where the same phoneme
        * codepoint is repeated for each id from that phoneme (usually just one id
        * plus pad).
        *
        * Groups of repeated codepoints are separated by a 0 so that alignments can
        * be attributed to the correct phoneme. This is accomplished by:
        *
        * 1. Read N (repeated) codepoints from phonemes until a 0 is reached (or end)
        * 2. The next N phoneme ids correspond to that phoneme
        * 3. The next N alignments (sample counts) correspond to that phoneme
        * 4. Advance your iterators in the phoneme id and alignment arrays by N
        * 5. Repeat
        */
        std::span<const char32_t> phonemes{};

        /**
        * \brief Phoneme ids that produced this audio chunk.
        *
        * Ids will look like [1, 0, id1, 0, id2, 0, ..., 2] where:
        * 0 = pad
        * 1 = beginning of sentence
        * 2 = end of sentence
        */
        std::vector<int> phoneme_ids{};

        /**
        * \brief Audio sample count for each phoneme id.
        *
        * This includes the meta ids:
        * 0 = pad
        * 1 = beginning of sentence
        * 2 = end of sentence
        *
        * Use the phonemes array to align these sample counts with actual phonemes.
        */
        std::vector<int> alignments{};

        #endif
    };

    /**
    * \brief Text-to-speech synthesizer.
    */
    struct Synthesizer {
        // From config JSON file
        std::string espeak_voice = "en-us";
        int sample_rate;
        SpeakerId num_speakers;
        PhonemeIdMap phoneme_id_map;
        int hop_length = DEFAULT_HOP_LENGTH;

        SynthesizerOptions options {
            .speaker_id = 0,
            .length_scale = DEFAULT_LENGTH_SCALE,
            .noise_scale = DEFAULT_NOISE_SCALE,
            .noise_w_scale = DEFAULT_NOISE_W_SCALE
        };

        // onnx
        std::unique_ptr<Ort::Session> session;
        Ort::AllocatorWithDefaultOptions session_allocator;
        Ort::SessionOptions session_options;
        Ort::Env session_env;

        // synthesize state
        std::vector<std::pair<std::vector<Phoneme>, std::vector<PhonemeId>>> phoneme_id_queue;

        Synthesizer(const Synthesizer&) = delete;
        Synthesizer& operator=(const Synthesizer&) = delete;
        Synthesizer(Synthesizer&&) noexcept = delete;
        Synthesizer& operator=(Synthesizer&&) noexcept = delete;

        Synthesizer() = default;

    private:
        /**
        * \brief Start text-to-speech synthesis.
        *
        * \param synth Piper synthesizer.
        *
        * \param text text to synthesize into audio.
        *
        * \return true if success.
        */
        bool start(const std::string& text);
    public:

        /**
        * \brief Create a Piper text-to-speech synthesizer from a voice model.
        *
        * \param model_path path to ONNX voice model file.
        *
        * \param config_path path to JSON voice config file or NULL if it's the
        * model_path + .json.
        *
        * \param espeak_data_path path to the espeak-ng data
        * directory.
        *
        * \return a Piper text-to-speech synthesizer for the voice model.
        */
        static std::unique_ptr<Synthesizer> create(const std::string& model_path, const std::string& config_path, const std::string& espeak_data_path, std::optional<Ort::SessionOptions> options = std::nullopt);

        /**
        * \brief Synthesize audio.
        *
        * \param text text to synthesize into audio.
        *
        * \param callback callback invocable with const AudioChunk&, can optionally return false to stop early.
        *
        * \return true when complete or stopped early, false on error or otherwise.
        */
        template <std::invocable F>
        requires (std::invocable<F&, const AudioChunk&>)
        inline bool synthesize(const std::string& text, F&& callback) {
            if (!start(text)) return false;

            struct cleanup {
                Synthesizer* synth;
                ~cleanup() {
                    synth->phoneme_id_queue.clear();
                }
            } cleanup{this};

            const auto phoneme_id_queue_size = this->phoneme_id_queue.size();

            if (phoneme_id_queue_size == 0) {
                std::invoke(callback, AudioChunk{.is_last = true});
            }

            auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

            for (std::size_t i = 0; i < phoneme_id_queue_size; ++i) {
                // Process next list of phoneme ids
                auto& [next_phonemes, next_ids] = this->phoneme_id_queue[i];

                // Allocate
                std::array<int64_t, 1> phoneme_id_lengths{(int64_t)next_ids.size()};
                std::array<float, 3> scales{this->options.noise_scale, this->options.length_scale,
                                        this->options.noise_w_scale};

                std::vector<Ort::Value> input_tensors;
                input_tensors.reserve((this->num_speakers > 1) ? 5 : 4);
                const std::array<int64_t, 2> phoneme_ids_shape{1, (int64_t)next_ids.size()};
                input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
                    memoryInfo, next_ids.data(), next_ids.size(), phoneme_ids_shape.data(),
                    phoneme_ids_shape.size()));

                constexpr static std::array<int64_t, 1> phoneme_id_lengths_shape{
                    (int64_t)phoneme_id_lengths.size()};
                input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
                    memoryInfo, phoneme_id_lengths.data(), phoneme_id_lengths.size(),
                    phoneme_id_lengths_shape.data(), phoneme_id_lengths_shape.size()));

                static constexpr std::array<int64_t, 1> scales_shape{(int64_t)scales.size()};
                input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, scales.data(), scales.size(), scales_shape.data(),
                    scales_shape.size()));

                // Add speaker id.
                // NOTE: These must be kept outside the "if" below to avoid being
                // deallocated.
                std::array<int64_t, 1> speaker_id{(int64_t)this->options.speaker_id};
                static constexpr std::array<int64_t, 1> speaker_id_shape{(int64_t)speaker_id.size()};

                if (this->num_speakers > 1) {
                    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
                        memoryInfo, speaker_id.data(), speaker_id.size(),
                        speaker_id_shape.data(), speaker_id_shape.size()));
                }

                // From export_onnx.py
                static constexpr std::array<const char *, 4> input_names = {"input", "input_lengths",
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
                    return false;
                }

                auto audio_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
                const auto num_samples = audio_shape[audio_shape.size() - 1];

                const float *audio_tensor_data = output_tensors.front().GetTensorData<float>();
                AudioChunk chunk{
                    .samples = {audio_tensor_data, audio_tensor_data + num_samples},
                    .sample_rate = this->sample_rate,
                    .is_last = (i == (phoneme_id_queue_size - 1)),
                };

                #ifdef LIBPIPER_FULL_AUDIOCHUNK
                chunk.phonemes = next_phonemes;

                // Copy phoneme ids
                chunk.phoneme_ids.reserve(next_ids.size());
                for (auto phoneme_id : next_ids) {
                    if (phoneme_id < std::numeric_limits<int>::min() ||
                        phoneme_id > std::numeric_limits<int>::max()) {
                        continue;
                    }
                    chunk.phoneme_ids.push_back(static_cast<int>(phoneme_id));
                }

                // Check for alignments
                if (output_tensors.size() > 1) {
                    auto alignments_shape =
                        output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

                    const auto num_alignments = alignments_shape[alignments_shape.size() - 1];
                    const float *alignments_tensor_data =
                        output_tensors[1].GetTensorData<float>();

                    chunk.alignments.resize(num_alignments);
                    for (std::size_t i = 0; i < num_alignments; ++i) {
                        chunk.alignments[i] = (int)(alignments_tensor_data[i] * this->hop_length);
                    }
                }
                #endif

                std::invoke(callback, chunk);
            }

            return true;
        }

        ~Synthesizer();
    };
}
