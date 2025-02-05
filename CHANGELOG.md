# Changelog

## [0.18.1](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/compare/aleph-alpha-client-v0.18.0...aleph-alpha-client-v0.18.1) (2025-02-05)


### Bug Fixes

* TopLogprob is now public ([ede8003](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/ede8003f29b8c5a9121ee9fa86c84d60e77fd467))

## [0.18.0](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/compare/aleph-alpha-client-v0.17.0...aleph-alpha-client-v0.18.0) (2025-02-05)


### ⚠ BREAKING CHANGES

* Introduce Logprobs::Top
* Introduce Logprobs::Sampled

### Features

* Introduce Logprobs::Sampled ([15d7954](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/15d79540e7080b61afd2930bf6822e000c36e693))
* Introduce Logprobs::Top ([44b33bf](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/44b33bfdc9f3d22befd809070667000ed290ecce))

## [0.17.0](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/compare/aleph-alpha-client-v0.16.0...aleph-alpha-client-v0.17.0) (2025-02-04)


### ⚠ BREAKING CHANGES

* Separate  Sampling struct for chat

### Bug Fixes

* Panic if top_k is used with chat ([4e304d0](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/4e304d0c7cab4f5a53a6f5932511e8f4d277ab5a))
* Separate  Sampling struct for chat ([0431a04](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/0431a04866436cd9b28a028eebc63d4b88e842d9))

## [0.16.0](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/compare/aleph-alpha-client-v0.15.0...aleph-alpha-client-v0.16.0) (2025-02-04)


### ⚠ BREAKING CHANGES

* remove support for complete_with_one_of sampling param

### Features

* remove support for complete_with_one_of sampling param ([349cd16](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/349cd1677ca6883e495b7daf4fc0b15a57e1e0cb))
* support frequency penalty for completions ([de8874a](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/de8874ab1a4880c4485c5c36ef0dd02cdd33686a))
* support presence penalty for completions ([7105f7e](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/7105f7e758fac9bbaddf5fababb45cbb8c863641))
* support stop sequences on chat endpoint ([47bc954](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/47bc954226327ae8ab157cbb72469b9a0c4d5713))


### Bug Fixes

* serialize maximum tokens as max_tokens ([0491d1c](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/0491d1c20bc0db0c75ec72613e0698dfeffe7b29))

## [0.15.0](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/compare/aleph-alpha-client-v0.14.0...aleph-alpha-client-v0.15.0) (2024-12-10)


### ⚠ BREAKING CHANGES

* add option to ask for special tokens in completion response
* rename start_with to complete_with
* clean up naming of methods to setup clients
* token env is called PHARIA_AI_TOKEN
* replace from_authentication by new from_env method

### Features

* add option to ask for special tokens in completion response ([1dbcb77](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/1dbcb775fd89da53f9f0c032fb5366b87650a9e7))
* clean up naming of methods to setup clients ([5f18f38](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/5f18f3854835099f04fc505cdd38ef1cffe24a8a))
* rename base_url env variable to inference_url ([b9a2fd2](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/b9a2fd2ef9124542bca2f415d7288ab76781b4d9))
* rename start_with to complete_with ([edd5590](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/edd55901bf82247b31d1e051ed4eca388cb4d6cd))
* replace from_authentication by new from_env method ([d2da859](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/d2da859cedc6ec2865bc8875d6b7dde2311238a5))
* token env is called PHARIA_AI_TOKEN ([827b44c](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/827b44c0e39168e8caa10acd90970872645528e8))

## [0.14.0](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/compare/aleph-alpha-client-v0.13.2...aleph-alpha-client-v0.14.0) (2024-11-28)

### ⚠ BREAKING CHANGES

- Update `tokenizers` dependency to v0.21.0 [ed5ea41](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/ed5ea41f5be7f3cc50d1edf86197ee8f5f0ed4c9)

## [0.13.2](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/compare/aleph-alpha-client-v0.13.1...aleph-alpha-client-v0.13.2) (2024-10-28)

### Features

- add stream completion method ([c513a36](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/c513a3651879a337e5c8f64abd1a769d067f1f36))

## [0.13.1](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/compare/aleph-alpha-client-v0.13.0...aleph-alpha-client-v0.13.1) (2024-10-24)

### Features

- add id to release workflow ([4fb33c2](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/4fb33c20d90f1231bacff148dd15ed69c2c7b1ed))

## [0.13.0](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/compare/aleph-alpha-client-v0.12.3...aleph-alpha-client-v0.13.0) (2024-10-24)

### ⚠ BREAKING CHANGES

- do not hide role behind enum

### Features

- add with_messages method to construct TaskChat ([8fc5c10](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/8fc5c10e4b24c351c70c16d0d7405e2a5cc0f4f8))
- do not hide role behind enum ([6ef29c5](https://github.com/Aleph-Alpha/aleph-alpha-client-rs/commit/6ef29c5d74824b801850021d3c3be550f0502057))

## 0.12.3

- Make `ChatOutput` public

## 0.12.2

- Fix missing import in doc string

## 0.12.1

- Add `Client::chat` method to send chat messages to a model

## 0.12.0

- Add `Client::tokenizer_by_model` to fetch the Tokenizer for a given model name

## 0.11.0

- Add `with_maximum_tokens` method to `Prompt`
- Remove maximum tokens argument from `Prompt::from_text`
- Make maximum tokens optional

## 0.10.1

- Fix: Version number in Cargo.toml

## 0.10.0

- Add the option to have authentication exclusively on a per request basis, without the need to specify a dummy token.
- Rename `Client::new` to `Client::with_authentication`.

## 0.9.0

- Add `How::api_token` to allow specifying API tokens for individual requests.

## 0.8.0

- Add `Error::Unavailable` to decouple service unavailability from 'queue full' 503 responses.

## 0.7.1

- Add `Client::tokenize` and `Client::detokenize`. Thanks to @andreaskoepf

## 0.7.0

- Add `client_timeout` to `How`
- Remove builder-methods from `How` as it introduced an unnecessary level of indirection

## 0.6.1

- Add `explanation` to `Client` for submitting explanation requests to the API
- Add `be_nice`-builder-method to `How` to make maintaining backwards compatibility easier

## 0.6.0

- Add `start_with_one_of` option to `Sampling`, to force the model to start a completion with one of several options.

## 0.5.7

- Add new `Prompt` method `join_consecutive_text_items` to make it easier to construct few shot prompts and other such use cases programmatically, without introducing strange tokenization side effects.

## 0.5.6

- Allow for executing `TaskSemanticEmbedding` without specifying models.

## 0.5.5

- Support for processing images already in memory via `Modality::from_image`.

## 0.5.4

- `Modality::from_image_path` now works with string literals.

## 0.5.3

- Fix version number

## 0.5.2

- Preprocess image on client side

## 0.5.1

- Minimal support for sending multimodal prompts

## 0.5.0

- Removed deprecated function `Client::complete`. It has been replaced with `Client::execute`.
- Introduced `how` parameter to `Client::execute` in order to control whether the request has the nice flag set, or not.

## 0.4.1

- Allow for `Prompt::from_text` to be called with `String` or any other type which implement `Into<Cow<'_, str>>`. This enables prompts to take ownership of their values which is practical for use cases there you want to return prompts from functions, which are based on locally generated strings.

## 0.4.0

- Support for stop sequences

## 0.3.0

- Add `TaskSemanticEmbedding`.
- `Completion` renamed to `CompletionOutput`

## 0.2.0

- Add `Error::Busy` to conveniently handle busy models.

## 0.1.1

- Fix: `Client::new` did not work due to a missing `https://` in the base URL.

## 0.1.0

- Initial release allows for sending receiving simple (not all parameters of the HTTP API are supported) completion requests.
