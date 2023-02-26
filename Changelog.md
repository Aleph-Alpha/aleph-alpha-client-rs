# Changelog

## 0.5.3

* Fix version number

## 0.5.2

* Preprocess image on client side

## 0.5.1

* Minimal support for sending multimodal prompts

## 0.5.0

* Removed deprecated function `Client::complete`. It has been replaced with `Client::execute`.
* Intruduced `how` parameter to `Client::execute` in order to control wether the request has the nice flag set, or not.

## 0.4.1

* Allow for `Prompt::from_text` to be called with `String` or any other type which implement `Into<Cow<'_, str>>`. This enables prompts to take ownership of their values which is practical for usecases there you want to return prompts from functions, which are based on locally generated strings.

## 0.4.0

* Support for stop sequences

## 0.3.0

* Add `TaskSemanticEmbedding`.
* `Completion` renamed to `CompletionOutput`

## 0.2.0

* Add `Error::Busy` to conviniently handle busy models.

## 0.1.1

* Fix: `Client::new` did not work due to a missing `https://` in the base URL.

## 0.1.0

* Initial release allows for sending receiving simple (not all parameters of the http API are supported) completion requests.
