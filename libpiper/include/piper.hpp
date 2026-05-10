#pragma once

#include <span>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <stdint.h>
#include <string>
#include <vector>

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
    constexpr inline int PIPER_OK = 0;
    constexpr inline int PIPER_DONE = 1;
    constexpr inline int PIPER_ERR_GENERIC = -1;

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
        std::span<const float> samples;

        /**
        * \brief Sample rate in Hertz.
        */
        int sample_rate;

        /**
        * \brief True if this is the last audio chunk.
        */
        bool is_last;

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
        std::span<const char32_t> phonemes;

        /**
        * \brief Phoneme ids that produced this audio chunk.
        *
        * Ids will look like [1, 0, id1, 0, id2, 0, ..., 2] where:
        * 0 = pad
        * 1 = beginning of sentence
        * 2 = end of sentence
        */
        std::span<const int> phoneme_ids;

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
        std::span<const int> alignments;
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
        std::queue<std::pair<std::vector<Phoneme>, std::vector<PhonemeId>>> phoneme_id_queue;
        std::vector<float> chunk_samples;
        std::vector<int> chunk_phoneme_ids;
        std::vector<Phoneme> chunk_phonemes;
        std::vector<int> chunk_alignments;

        Synthesizer(const Synthesizer&) = delete;
        Synthesizer& operator=(const Synthesizer&) = delete;
        Synthesizer(Synthesizer&&) noexcept = delete;
        Synthesizer& operator=(Synthesizer&&) noexcept = delete;

        Synthesizer() = default;

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
        * \brief Start text-to-speech synthesis.
        *
        * \param synth Piper synthesizer.
        *
        * \param text text to synthesize into audio.
        *
        * \sa \ref Synthesizer::next
        *
        * \return PIPER_OK or error code.
        */
        int start(const std::string& text);

        /**
        * \brief Synthesize next chunk of audio.
        *
        * \param synth Piper synthesizer.
        *
        * \param chunk audio chunk to fill.
        *
        * piper_synthesize_start must be called before this function.
        * Each call to piper_synthesize_next will fill the audio chunk, invalidating
        * the memory of the previous chunk.
        * The final audio chunk will have is_last = true.
        * A return value of PIPER_DONE indicates that synthesis is complete.
        *
        * \sa \ref Synthesizer::start
        *
        * \return PIPER_DONE when complete, otherwise PIPER_OK or error code.
        */
        int next(AudioChunk& chunk);

        ~Synthesizer();
    };
}
