# Changelog

## Unreleased

* Add `Error::Unavailable` to decouple service unavailability from 'queue full' 503 responses.

## 0.7.1

* Add `Client::tokenize` and `Client::detokenize`. Thanks to @andreaskoepf

## 0.7.0

* Add `client_timeout` to `How`
* Remove builder-methods from `How` as it introduced an unnecessary level of indirection

## 0.6.1

* Add `explanation` to `Client` for submitting explanation requests to the API
* Add `be_nice`-builder-method to `How` to make maintaining backwards compatibility easier

## 0.6.0

* Add `start_with_one_of` option to `Sampling`, to force the model to start a completion with one of several options.

## 0.5.7

* Add new `Prompt` method `join_consecutive_text_items` to make it easier to construct few shot prompts and other such use cases programmatically, without introducing strange tokenization side effects.

## 0.5.6

* Allow for executing `TaskSemanticEmbedding` without specifying models.

## 0.5.5

* Support for processing images already in memory via `Modality::from_image`.

## 0.5.4

* `Modality::from_image_path` now works with string literals.

## 0.5.3

* Fix version number

## 0.5.2

* Preprocess image on client side

## 0.5.1

* Minimal support for sending multimodal prompts

## 0.5.0

* Removed deprecated function `Client::complete`. It has been replaced with `Client::execute`.
* Introduced `how` parameter to `Client::execute` in order to control whether the request has the nice flag set, or not.

## 0.4.1

* Allow for `Prompt::from_text` to be called with `String` or any other type which implement `Into<Cow<'_, str>>`. This enables prompts to take ownership of their values which is practical for use cases there you want to return prompts from functions, which are based on locally generated strings.

## 0.4.0

* Support for stop sequences

## 0.3.0

* Add `TaskSemanticEmbedding`.
* `Completion` renamed to `CompletionOutput`

## 0.2.0

* Add `Error::Busy` to conveniently handle busy models.

## 0.1.1

* Fix: `Client::new` did not work due to a missing `https://` in the base URL.

## 0.1.0

* Initial release allows for sending receiving simple (not all parameters of the HTTP API are supported) completion requests.
