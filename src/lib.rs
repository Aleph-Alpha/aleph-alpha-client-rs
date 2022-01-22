use reqwest::{header, ClientBuilder};

/// Sends HTTP request to the Aleph Alpha API
pub struct Client {
    pub base: String,
    pub http: reqwest::Client,
}

impl Client {
    /// In production you typically would want this to "https://api.aleph-alpha.de". Yet you may
    /// want to use different instances for testing.
    pub fn with_base_uri(base: String, token: &str) -> Self {
        let mut headers = header::HeaderMap::new();

        let mut auth_value = header::HeaderValue::from_str(&format!("Bearer {}", token)).unwrap();
        // Consider marking security-sensitive headers with `set_sensitive`.
        auth_value.set_sensitive(true);
        headers.insert(header::AUTHORIZATION, auth_value);

        let http = ClientBuilder::new()
            .default_headers(headers)
            .build()
            .unwrap();
            
        Self { base, http }
    }

    pub async fn complete(&self, task: String) -> String {
        self.http
            .post(format!("{}/complete", self.base))
            .header("Content-Type", "application/json")
            .body(task)
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
    }
}
