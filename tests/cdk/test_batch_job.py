from hls_constructs import ecr_uri_to_repo_arn


def test_ecr_uri_to_repo_arn_private():
    uri = "012345678901.dkr.ecr.us-west-2.amazonaws.com/hls-composites:v1.2.3"
    assert (
        ecr_uri_to_repo_arn(uri)
        == "arn:aws:ecr:us-west-2:012345678901:repository/hls-composites"
    )


def test_ecr_uri_to_repo_arn_namespaced_repo():
    uri = "012345678901.dkr.ecr.us-west-2.amazonaws.com/team/hls-composites:latest"
    assert (
        ecr_uri_to_repo_arn(uri)
        == "arn:aws:ecr:us-west-2:012345678901:repository/team/hls-composites"
    )


def test_ecr_uri_to_repo_arn_public_is_none():
    assert ecr_uri_to_repo_arn("public.ecr.aws/amazonlinux/amazonlinux:latest") is None
