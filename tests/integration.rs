use aleph_alpha_client::{Authentication, Client, Error, Prompt, Sampling, TaskCompletion};
use wiremock::{
    matchers::{body_json_string, header, method, path},
    Mock, MockServer, ResponseTemplate,
};

#[tokio::test]
async fn completion_with_luminous_base() {
    // Given

    // Start a background HTTP server on a random local part
    let mock_server = MockServer::start().await;

    let answer = r#"{"model_version":"2021-12","completions":[{"completion":"\n","finish_reason":"maximum_tokens"}]}"#;
    let body = r#"{
        "model": "luminous-base",
        "prompt": [{"type": "text", "data": "Hello,"}],
        "maximum_tokens": 1
    }"#;

    Mock::given(method("POST"))
        .and(path("/complete"))
        .and(header("Authorization", "Bearer dummy-token"))
        .and(header("Content-Type", "application/json"))
        .and(body_json_string(body))
        .respond_with(ResponseTemplate::new(200).set_body_string(answer))
        // Mounting the mock on the mock server - it's now effective!
        .mount(&mock_server)
        .await;

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_text("Hello,"),
        maximum_tokens: 1,
        sampling: Sampling::Deterministic,
    };

    let model = "luminous-base";

    let client = Client::with_base_url(mock_server.uri(), Authentication::ApiToken("dummy-token"))
        .await
        .unwrap();
    let response = client.complete(model, &task).await.unwrap();
    let actual = response.best_text();

    // Then
    assert_eq!("\n", actual)
}

/// If we open too many tasks at once, at one point the API, will tell us that we exceeded our
/// quota. This might change very dynamically, depending on how much compute is available and
/// how many other users are trying to utilize the API.
///
/// It should be easy for client code to detect this scenario.
#[tokio::test]
async fn detect_too_many_tasks() {
    // Given

    // Start a background HTTP server on a random local part
    let mock_server = MockServer::start().await;

    let answer =
        r#"{"error":"There are already running too many tasks for you","code":"TOO_MANY_TASKS"}"#;
    let body = r#"{
        "model": "luminous-base",
        "prompt": [{"type": "text", "data": "Hello,"}],
        "maximum_tokens": 1
    }"#;

    Mock::given(method("POST"))
        .and(path("/complete"))
        .and(header("Authorization", "Bearer dummy-token"))
        .and(header("Content-Type", "application/json"))
        .and(body_json_string(body))
        .respond_with(ResponseTemplate::new(429).set_body_string(answer))
        // Mounting the mock on the mock server - it's now effective!
        .mount(&mock_server)
        .await;

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_text("Hello,"),
        maximum_tokens: 1,
        sampling: Sampling::Deterministic,
    };

    let model = "luminous-base";

    let client = Client::with_base_url(mock_server.uri(), Authentication::ApiToken("dummy-token"))
        .await
        .unwrap();
    let error = client.complete(model, &task).await.unwrap_err();

    assert!(matches!(error, Error::TooManyTasks));
}

/// If we open too many requests at once, we may trigger rate limmiting. We want this scenario to be
/// easily detectible by the user, so he/she/it can start sending requests slower.
#[tokio::test]
async fn detect_rate_limmiting() {
    // Given

    // Start a background HTTP server on a random local part
    let mock_server = MockServer::start().await;

    let answer = r#"Too many requests"#;
    let body = r#"{
        "model": "luminous-base",
        "prompt": [{"type": "text", "data": "Hello,"}],
        "maximum_tokens": 1
    }"#;

    Mock::given(method("POST"))
        .and(path("/complete"))
        .and(header("Authorization", "Bearer dummy-token"))
        .and(header("Content-Type", "application/json"))
        .and(body_json_string(body))
        .respond_with(ResponseTemplate::new(429).set_body_string(answer))
        // Mounting the mock on the mock server - it's now effective!
        .mount(&mock_server)
        .await;

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_text("Hello,"),
        maximum_tokens: 1,
        sampling: Sampling::Deterministic,
    };

    let model = "luminous-base";

    let client = Client::with_base_url(mock_server.uri(), Authentication::ApiToken("dummy-token"))
        .await
        .unwrap();
    let error = client.complete(model, &task).await.unwrap_err();

    assert!(matches!(error, Error::TooManyRequests));
}

#[tokio::test]
async fn login_with_invalid_credentials() {
    // Given

    // Start a background HTTP server on a random local part
    let mock_server = MockServer::start().await;

    let body = r#"{"email":"user@corp.com", "password":"Invalid"}"#;
    let response = r#"{"error":"Invalid username or password","code":"INVALID_CREDENTIALS"}"#;

    Mock::given(method("POST"))
        .and(path("/users/login"))
        .and(header("Content-Type", "application/json"))
        .and(body_json_string(body))
        .respond_with(ResponseTemplate::new(401).set_body_string(response))
        // Mounting the mock on the mock server - it's now effective!
        .mount(&mock_server)
        .await;

    // When user tries to acquire token
    let result = Client::with_base_url(
        mock_server.uri(),
        Authentication::Credentials {
            user: "user@corp.com",
            password: "MySecret",
        },
    )
    .await;

    // Then
    assert!(result.is_err())
}

#[tokio::test]
async fn login_with_credentials() {
    // Given

    // Start a background HTTP server on a random local part
    let mock_server = MockServer::start().await;

    let body = r#"{"email":"user@corp.com","password":"MySecret"}"#;
    //let response = r#"{"id":12,"email":"user@corp.com","role":"Client","token":"dummy-token","first_failed_login":"2022-02-16T13:14:17.328878977Z","num_failed_logins":1,"locked":false}"#;
    let response = r#"{"token":"dummy-token"}"#;

    Mock::given(method("POST"))
        .and(path("/users/login"))
        .and(header("Content-Type", "application/json"))
        .and(body_json_string(body))
        .respond_with(ResponseTemplate::new(200).set_body_string(response))
        // Mounting the mock on the mock server - it's now effective!
        .mount(&mock_server)
        .await;

    // When user tries to acquire token
    let result = Client::with_base_url(
        mock_server.uri(),
        Authentication::Credentials {
            user: "user@corp.com",
            password: "MySecret",
        },
    )
    .await;

    // Then
    assert!(result.is_ok())
}
