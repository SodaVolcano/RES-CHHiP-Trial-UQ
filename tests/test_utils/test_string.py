from tests.context import capture_placeholders, placeholder_matches


class TestCapturePlaceholders:

    # correctly replaces placeholders with default regex pattern
    def test_correctly_replaces_placeholders_with_default_regex_pattern(self):
        s = "somestuff{a}_{b}adf{c}."
        placeholders = ["a", "b"]
        expected = "somestuff(.*?)_(.*?)adf.*?\\."

        result = capture_placeholders(s, placeholders, r".*?")
        assert result == expected

    # handles empty string input
    def test_handles_empty_string_input(self):
        s = ""
        placeholders = ["a", "b"]
        expected = ""

        result = capture_placeholders(s, placeholders)
        assert result == expected

    # handles strings with no placeholders gracefully
    def test_handles_strings_with_no_placeholders_gracefully(self):
        s = "no placeholders here"
        placeholders = []
        expected = "no\\ placeholders\\ here"

        result = capture_placeholders(s, placeholders)
        assert result == expected

    # handles empty list of placeholders
    def test_handles_empty_list_of_placeholders(self):
        s = "somestuff{a}_{b}.nii.gz"
        placeholders = []
        expected = "somestuff.*?_.*?\\.nii\\.gz"

        result = capture_placeholders(s, placeholders, r".*?")
        assert result == expected


class TestPlaceholderMatches:

    # Matches placeholders correctly in a list of strings
    def test_matches_placeholders_correctly(self):
        str_list = [
            "/path/to/eye_sp.nii.gz",
            "/path/to/bladder_jd.nii.gz",
            "/path/to/brain_md.nii.gz",
        ]
        pattern = "/path/to/{organ}_{observer}.nii.gz"
        placeholders = ["organ", "observer"]
        expected_output = [("eye", "sp"), ("bladder", "jd"), ("brain", "md")]

        result = placeholder_matches(str_list, pattern, placeholders)

        assert all([x in expected_output for x in result])

    # Handles empty list of strings
    def test_handles_empty_list_of_strings(self):
        str_list = []
        pattern = "/path/to/{organ}_{observer}.nii.gz"
        placeholders = ["organ", "observer"]
        expected_output = []

        result = placeholder_matches(str_list, pattern, placeholders)

        assert result == expected_output

    # Handles empty list of placeholders
    def test_handles_empty_list_of_placeholders(self):
        str_list = [
            "/path/to/bladder_jd.nii.gz",
            "/path/to/brain_md.nii.gz",
            "/path/to/eye_sp.nii.gz",
        ]
        pattern = "/path/to/{organ}_{observer}.nii.gz"
        placeholders = []
        expected_output = [(), (), ()]

        result = placeholder_matches(str_list, pattern, placeholders)

        assert result == expected_output

    # Handles patterns with placeholders not present in the strings
    def test_handles_patterns_with_placeholders_not_present(self):
        str_list = [
            "/path/to/bladder_jd.nii.gz",
            "/path/to/brain_md.nii.gz",
            "/path/to/eye_sp.nii.gz",
        ]
        pattern = "to/{organ}_{observer}_{a}.nii.gz"
        placeholders = ["organ", "observer", "a"]
        re_pattern = r".+"
        expected_output = []

        result = placeholder_matches(str_list, pattern, placeholders, re_pattern)

        assert result == expected_output

    # Handles empty pattern string
    def test_handles_empty_pattern_string(self):
        str_list = [
            "/path/to/bladder_jd.nii.gz",
            "/path/to/brain_md.nii.gz",
            "/path/to/eye_sp.nii.gz",
        ]
        pattern = ""
        placeholders = ["organ", "observer"]
        expected_output = [(), (), ()]

        result = placeholder_matches(str_list, pattern, placeholders)

        assert result == expected_output
