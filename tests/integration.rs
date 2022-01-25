use aleph_alpha_client::{Client, CompletionBody, Prompt};
use wiremock::{
    matchers::{body_json_string, header, method, path},
    Mock, MockServer, ResponseTemplate,
};

#[tokio::test]
async fn completion_with_luminous_base() {
    // Given

    // Start a background HTTP server on a random local part
    let mock_server = MockServer::start().await;

    let token = "dummy-token";
    let answer = r#"{"id":"273a3698-876e-49cb-af71-dbe7a249df92","model_version":"2021-12","completions":[{"completion":"\n","finish_reason":"maximum_tokens"}]}"#;
    let body = r#"{
        "model": "luminous-base",
        "prompt": [{"type": "text", "data": "Hello,"}],
        "maximum_tokens": 1
    }"#;

    Mock::given(method("POST"))
        .and(path("/complete"))
        .and(header("Authorization", format!("Bearer {token}").as_str()))
        .and(header("Content-Type", "application/json"))
        .and(body_json_string(body))
        .respond_with(ResponseTemplate::new(200).set_body_string(answer))
        // Mounting the mock on the mock server - it's now effective!
        .mount(&mock_server)
        .await;

    // When
    let task = CompletionBody {
        model: "luminous-base",
        prompt: Prompt::from_text("Hello,"),
        maximum_tokens: 1,
    };

    let client = Client::with_base_uri(mock_server.uri(), token).unwrap();
    let response = client.complete(&task).await.unwrap();

    // Then
    eprintln!("{}", response);
    assert_eq!(answer, response)
}
