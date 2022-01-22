use reqwest::ClientBuilder;

/// Sends HTTP request to the Aleph Alpha API
pub struct Client {
    pub base: String,
    pub token: String,
    pub http: reqwest::Client,
}

impl Client {
    /// In production you typically would want this to "https://api.aleph-alpha.de". Yet you may
    /// want to use different instances for testing.
    pub fn with_base_uri(base: String, token: String) -> Self {
        let http = ClientBuilder::new().build().unwrap();
        Self { base, token, http }
    }

    pub async fn complete(&self, task: String) -> String {
        self.http
            .post(format!("{}/complete", self.base))
            .header("Authorization", format!("Bearer {}", self.token))
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
