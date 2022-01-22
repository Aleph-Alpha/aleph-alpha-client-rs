#[cfg(test)]
mod tests {
    use reqwest::Client;
    use wiremock::{
        matchers::{header, method, path},
        Mock, MockServer, ResponseTemplate,
    };

    #[tokio::test]
    async fn completion_with_luminous_base() {
        // Start a background HTTP server on a random local part
        let mock_server = MockServer::start().await;

        let token = "dummy-token";
        let answer = r#"{"id":"273a3698-876e-49cb-af71-dbe7a249df92","model_version":"2021-12","completions":[{"completion":"\n","finish_reason":"maximum_tokens"}]}"#;

        Mock::given(method("POST"))
            .and(path("/complete"))
            .and(header(
                "Authorization",
                format!("Bearer {}", token).as_str(),
            ))
            .and(header("Content-Type", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_string(answer))
            // Mounting the mock on the mock server - it's now effective!
            .mount(&mock_server)
            .await;

        let body = r#"{
            "model": "luminous-base",
            "prompt": [{"type": "text", "data": "Hello,"}],
            "maximum_tokens": 1
        }"#;

        let client = Client::new();
        let response = client
            .post(format!("{}/complete", mock_server.uri()))
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await
            .unwrap();

        let body = response.text().await.unwrap();

        eprintln!("{}", body);

        assert_eq!(answer, body)
    }
}
