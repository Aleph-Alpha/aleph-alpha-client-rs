use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::Error;

#[derive(Clone, Copy)]
pub enum Authentication<'a> {
    /// Authenticate using username and password
    Credentials {
        /// Your username. Typically this is the email address you used to sign up. This is not case
        /// sensitive.
        user: &'a str,
        /// The password associated with your user.
        password: &'a str,
    },
    /// A permanent API Token used for authentication. Can be acquired by logging in using
    /// credentials and calling [`Self::api_token()`]
    ApiToken(&'a str),
}

impl<'a> Authentication<'a> {
    /// Either returns the internally stored token, or requests one from the API using the
    /// credentials.
    pub async fn api_token(&self, host: &str) -> Result<Cow<'a, str>, Error> {
        match self {
            Authentication::Credentials { user, password } => {
                let response = reqwest::Client::builder()
                    .build()?
                    .post(format!("{host}/users/login"))
                    .json(&LoginRequestBody {
                        email: user,
                        password,
                    })
                    .send()
                    .await?;

                let LoginResponseBody { token } = response.json().await?;

                Ok(Cow::Owned(token))
            }
            Authentication::ApiToken(token) => Ok(Cow::Borrowed(token)),
        }
    }
}

#[derive(Serialize)]
struct LoginRequestBody<'a> {
    email: &'a str,
    password: &'a str,
}

#[derive(Deserialize)]
struct LoginResponseBody {
    token: String,
}
