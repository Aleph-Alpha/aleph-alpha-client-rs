# Changelog

## 0.3.0

* Add `TaskSemanticEmbedding`.
* `Completion` renamed to `CompletionOutput`

## 0.2.0

* Add `Error::Busy` to conviniently handle busy models.

## 0.1.1

* Fix: `Client::new` did not work due to a missing `https://` in the base URL.

## 0.1.0

* Initial release allows for sending receiving simple (not all parameters of the http API are supported) completion requests.
