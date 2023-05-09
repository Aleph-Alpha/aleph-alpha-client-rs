use image::{
    imageops::FilterType::CatmullRom, DynamicImage, GenericImageView, ImageError, ImageFormat,
};
use std::{
    cmp::min,
    fs::File,
    io::{self, BufReader, Cursor},
    path::Path,
};
use thiserror::Error as ThisError;

/// Image is shrinked on the server side, before it is send to the model. We might as well save the
/// bandwith and do it right away.
const DESIRED_IMAGE_SIZE: u32 = 384;

pub fn from_image_path(path: &Path) -> Result<Vec<u8>, LoadImageError> {
    let file = BufReader::new(File::open(path).map_err(LoadImageError::Io)?);
    let format = ImageFormat::from_path(path).map_err(LoadImageError::UnknownImageFormat)?;
    let image = image::load(file, format).map_err(LoadImageError::InvalidImageEncoding)?;

    let bytes = preprocess_image(&image);
    Ok(bytes)
}

pub fn preprocess_image(org_image: &DynamicImage) -> Vec<u8> {
    let center_cropped = center_cropped(org_image);
    let resized = center_cropped.resize_exact(DESIRED_IMAGE_SIZE, DESIRED_IMAGE_SIZE, CatmullRom);
    let buf = Vec::new();
    let mut out = Cursor::new(buf);
    resized.write_to(&mut out, ImageFormat::Png).unwrap();
    out.into_inner()
}

fn center_cropped(image: &DynamicImage) -> DynamicImage {
    let (height, width) = image.dimensions();
    let size = min(height, width);
    let x = (height - size) / 2;
    let y = (width - size) / 2;
    image.crop_imm(x, y, width, height)
}

/// Errors returned by the Aleph Alpha Client
#[derive(ThisError, Debug)]
pub enum LoadImageError {
    #[error("Error decoding input image")]
    InvalidImageEncoding(#[source] ImageError),
    #[error("Failed to guess image format from path")]
    UnknownImageFormat(#[source] ImageError),
    #[error("Error opening input image file.")]
    Io(#[source] io::Error),
}
