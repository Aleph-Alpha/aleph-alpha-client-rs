use std::{
    borrow::{Borrow, Cow},
    path::Path,
};

use base64::{prelude::BASE64_STANDARD, Engine};
use image::DynamicImage;
use itertools::Itertools;
use serde::Serialize;

use crate::image_preprocessing::{self, LoadImageError};

/// A prompt which is passed to the model for inference. Usually it is one text item, but it could
/// also be a combination of several modalities like images and text.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct Prompt<'a>(Vec<Modality<'a>>);

impl<'a> Prompt<'a> {
    /// Create a prompt from a single text item.
    pub fn from_text(text: impl Into<Cow<'a, str>>) -> Self {
        Self(vec![Modality::from_text(text)])
    }

    /// Create a multimodal prompt from a list of individual items with any modality.
    pub fn from_vec(items: Vec<Modality<'a>>) -> Self {
        Self(items)
    }

    /// Allows you to borrow the contents of the prompt without allocating a new one.
    pub fn borrow(&'a self) -> Prompt<'a> {
        Self(self.0.iter().map(|item| item.borrow()).collect())
    }

    /// When constructing prompts programatically, it can be beneficial to append several
    /// text items in a prompt. For example, if doing a fewshot prompt as the first item,
    /// and user input as a second item.
    ///
    /// However, because of how tokenization works, having each item tokenized separately
    /// can sometimes have strange side effects (tokenizing two partial strings does not
    /// necessarily produce the same tokens as tokenizing the strings joined together).
    ///
    /// This method will take an existing prompt and merge any consecutive prompt items
    /// by a given separator. You can use an empty string for the separator if you want
    /// to just concatenate them.
    pub fn join_consecutive_text_items(&mut self, separator: &str) {
        self.0 = self
            .0
            .drain(..)
            .coalesce(|a, b| match (a, b) {
                (Modality::Text { mut data }, Modality::Text { data: other }) => {
                    data.to_mut().push_str(separator);
                    data.to_mut().push_str(&other);
                    Ok(Modality::Text { data })
                }
                (a, b) => Err((a, b)),
            })
            .collect::<Vec<_>>();
    }
}

/// The prompt for models can be a combination of different modalities (Text and Image). The type of
/// modalities which are supported depend on the Model in question.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Modality<'a> {
    /// The only type of prompt which can be used with pure language models
    Text { data: Cow<'a, str> },
    /// An image input into the model. See [`Modality::from_image_path`].
    Image { data: Cow<'a, str> },
}

impl<'a> Modality<'a> {
    /// Instantiates a text prompt
    pub fn from_text(text: impl Into<Cow<'a, str>>) -> Self {
        Modality::Text { data: text.into() }
    }

    /// Image input for model, from file path.
    ///
    /// The model can only see squared pictures. Images are centercropped.
    ///
    /// ```no_run
    /// use aleph_alpha_client::{Client, How, Modality, Prompt, Sampling, Stopping, TaskCompletion, Task};
    /// use dotenv::dotenv;
    /// use std::path::PathBuf;
    ///
    /// #[tokio::main(flavor = "current_thread")]
    /// async fn main() {
    ///     // Create client
    ///     let client = Client::from_env().unwrap();
    ///     // Define task
    ///     let task = TaskCompletion {
    ///         prompt: Prompt::from_vec(vec![
    ///             Modality::from_image_path("cat.png").unwrap(),
    ///             Modality::from_text("A picture of "),
    ///         ]),
    ///         stopping: Stopping::from_maximum_tokens(10),
    ///         sampling: Sampling::MOST_LIKELY,
    ///     };
    ///     // Execute
    ///     let model = "luminous-base";
    ///     let job = task.with_model(model);
    ///     let response = client.output_of(&job, &How::default()).await.unwrap();
    ///     // Show result
    ///     println!("{}", response.completion);
    /// }
    /// ```
    pub fn from_image_path(path: impl AsRef<Path>) -> Result<Self, LoadImageError> {
        let bytes = image_preprocessing::from_image_path(path.as_ref())?;
        Ok(Self::from_image_bytes(&bytes))
    }

    /// Image input for model
    ///
    /// The model can only see squared pictures. Images are centercropped. You may want to use this
    /// method instead of [`Self::from_image_path`] in case you have the image in memory already
    /// and do not want to load it from a file again.
    pub fn from_image(image: &DynamicImage) -> Result<Self, LoadImageError> {
        let bytes = image_preprocessing::preprocess_image(image);
        Ok(Self::from_image_bytes(&bytes))
    }

    /// Generates an image input from the binary representation of the image.
    ///
    /// Using this constructor you must use a binary representation compatible with the API. Png is
    /// guaranteed to be supported, and all others formats are converted into it. Furthermore, the
    /// model can only look at square shaped pictures. If the picture is not square shaped it will
    /// be center cropped.
    fn from_image_bytes(image: &[u8]) -> Self {
        Modality::Image {
            data: BASE64_STANDARD.encode(image).into(),
        }
    }

    /// Create a semantically idetical entry of modality which borrows the contents of this one.
    ///
    /// It is very practical to allow Modality of e.g. Text to take both ownership of the string it
    /// contains as well as borrow a slice. However then we are creating a body from the user input
    /// we want to avoid copying everything and needing to allocate for that modality again. This is
    /// there this borrow function really shines.
    pub fn borrow(&self) -> Modality<'_> {
        match self {
            Modality::Text { data } => Modality::Text {
                data: Cow::Borrowed(data.borrow()),
            },
            Modality::Image { data } => Modality::Image {
                data: Cow::Borrowed(data.borrow()),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_concatenate_prompt_items() {
        let mut prompt =
            Prompt::from_vec(vec![Modality::from_text("foo"), Modality::from_text("bar")]);
        prompt.join_consecutive_text_items("");

        assert_eq!(prompt.0, vec![Modality::from_text("foobar")]);
    }

    #[test]
    fn can_concatenate_prompt_items_with_custom_separator() {
        let mut prompt =
            Prompt::from_vec(vec![Modality::from_text("foo"), Modality::from_text("bar")]);
        prompt.join_consecutive_text_items("\n");

        assert_eq!(prompt.0, vec![Modality::from_text("foo\nbar")]);
    }
}
