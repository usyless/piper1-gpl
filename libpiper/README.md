# 🔧 Piper C++ API

Include in your own cmake project:

```cmake
FetchContent_Declare(
    libpiper
    GIT_REPOSITORY https://github.com/usyless/piper1-gpl.git
    GIT_TAG        main
    SOURCE_SUBDIR  libpiper
)

FetchContent_MakeAvailable(libpiper)
```

This will automatically download/build [espeak-ng][] as well as download shared libraries for the [onnxruntime][].

Then you want to include `<piper.hpp>`

To move onnx and espeak-ng data you can use

```cmake
file(GLOB ONNX_LIB_PATH "${CMAKE_CURRENT_BINARY_DIR}/_deps/libpiper-src/libpiper/lib/onnxruntime-*/lib/libonnxruntime.so.1")

add_custom_command(TARGET ${PROJECT_NAME}
                POST_BUILD

                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                        "${ONNX_LIB_PATH}"
                        "${CMAKE_CURRENT_BINARY_DIR}"

                COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
                        "${CMAKE_CURRENT_BINARY_DIR}/_deps/espeak_ng_external-build/espeak-ng-data"
                        "${CMAKE_CURRENT_BINARY_DIR}/espeak-data"
)
```

## Example

```c++
#include <fstream>
#include <piper.hpp>

int main() {
    auto synth = piper::Synthesizer::create("/path/to/voice.onnx",
                                            "/path/to/voice.onnx.json",
                                            "/path/to/espeak-ng-data");
    
    if (!synth) return 1;

    // aplay -r 22050 -c 1 -f FLOAT_LE -t raw output.raw
    std::ofstream audio_stream("output.raw", std::ios::binary);

    // Change options under synth->options before calling start

    synth->options.speaker_id = 1;

    if (synth->start("Random speech text") != piper::PIPER_OK) return 1;

    piper::AudioChunk chunk;
    while (synth->next(chunk) != piper::PIPER_DONE) {
        audio_stream.write(reinterpret_cast<const char *>(chunk.samples.data()),
                           chunk.samples.size() * sizeof(float));
    }

    // synth automatically cleared up when it goes out of scope
    return 0;
}
```

<!-- Links -->
[espeak-ng]: https://github.com/espeak-ng/espeak-ng
[onnxruntime]: https://github.com/microsoft/onnxruntime
