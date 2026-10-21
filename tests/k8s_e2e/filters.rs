use crate::fixtures::{chat_completion, ensure_gateway_ready, gateway_url, http_client};

#[tokio::test]
async fn praxis_version_header_applied() {
    ensure_gateway_ready().await;
    let resp = chat_completion("gpt-4", "hello").await;

    assert_eq!(resp.status(), 200);

    let version = resp
        .headers()
        .get("X-Praxis-Version")
        .expect("missing X-Praxis-Version header")
        .to_str()
        .expect("header value not valid UTF-8");
    assert_eq!(version, "e2e");
}

/// Send a chat completion to `path` carrying an `X-Maas-Username` header,
/// on top of `http_client()`'s default `Authorization` header.
async fn completion_with_identity_headers(path: &str) -> reqwest::Response {
    let client = http_client();
    let url = format!("{}{path}", gateway_url());

    client
        .post(&url)
        .header("X-Maas-Username", "e2e-test-user")
        .json(&serde_json::json!({
            "model": "granite-8b",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .expect("request failed")
}

/// maas-headers-guard (opendatahub-io/praxis-extproc#160): the e2e
/// `ai-filters` chain mirrors the `ipp` chain's guard filter, which must
/// strip `authorization` and `x-maas-*` headers from inference requests
/// before they reach the upstream model.
///
/// Uses the dedicated `/direct/auth-guard` path rather than the shared
/// `/v1/chat/completions` path: the latter also carries the pre-existing
/// `gpt-4` tests, which route to the real external llm-katan backend and
/// need their `Authorization` header (`fixtures::http_client()`'s default)
/// intact to authenticate there. See
/// deploy/overlays/e2e/workload/configmap-patch.yaml for why the guard's
/// `authorization`/`x-api-key` removal is scoped to this path instead of
/// (as first tried) a model-name header condition.
#[tokio::test]
async fn maas_headers_guard_strips_on_inference_path() {
    ensure_gateway_ready().await;
    let resp = completion_with_identity_headers("/direct/auth-guard/v1/chat/completions").await;

    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers().get("X-Guard-Leaked-Authorization").is_none(),
        "authorization header reached the upstream on an inference path"
    );
    assert!(
        resp.headers().get("X-Guard-Leaked-X-Maas-Username").is_none(),
        "x-maas-username header reached the upstream on an inference path"
    );
}

/// The guard must exempt maas-api's own (non-inference) routes, so a
/// request under /maas-api keeps its authorization / x-maas-* headers.
#[tokio::test]
async fn maas_headers_guard_exempts_maas_api_path() {
    ensure_gateway_ready().await;
    let resp = completion_with_identity_headers("/maas-api/v1/chat/completions").await;

    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers().get("X-Guard-Leaked-Authorization").is_some(),
        "authorization header was stripped on an exempted maas-api path"
    );
    assert!(
        resp.headers().get("X-Guard-Leaked-X-Maas-Username").is_some(),
        "x-maas-username header was stripped on an exempted maas-api path"
    );
}
