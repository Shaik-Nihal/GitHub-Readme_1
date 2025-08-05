import unittest
from readme_generator import parse_github_url

class TestReadmeGenerator(unittest.TestCase):

    def test_parse_github_url_valid(self):
        self.assertEqual(parse_github_url("https://github.com/owner/repo"), ("owner", "repo"))
        self.assertEqual(parse_github_url("http://github.com/owner/repo-name_with-hyphens.and.dots"), ("owner", "repo-name_with-hyphens.and.dots"))
        self.assertEqual(parse_github_url("https://www.github.com/some_Owner/some_Repo/"), ("some_Owner", "some_Repo")) # With trailing slash
        self.assertEqual(parse_github_url("github.com/user/project"), ("user", "project")) # No scheme
        self.assertEqual(parse_github_url("HTTPS://GITHUB.COM/TestUser/TestRepo"), ("TestUser", "TestRepo")) # Case insensitivity for domain

    def test_parse_github_url_invalid(self):
        # Updated to check for the beginning of the new error message
        invalid_message_regex = r"Invalid or unsupported GitHub URL format:.*"
        with self.assertRaisesRegex(ValueError, invalid_message_regex):
            parse_github_url("https://example.com/owner/repo")
        with self.assertRaisesRegex(ValueError, invalid_message_regex):
            parse_github_url("https://github.com/owner") # Missing repo
        with self.assertRaisesRegex(ValueError, invalid_message_regex):
            parse_github_url("https://github.com/") # Missing owner and repo
        with self.assertRaisesRegex(ValueError, invalid_message_regex):
            parse_github_url("just_a_string")
        with self.assertRaisesRegex(ValueError, invalid_message_regex):
            parse_github_url("http://github.com//repo") # Empty owner
        # The following case might now be caught differently or pass if "owner" is valid and "" is repo, then "" fails.
        # Let's test the specific error for "owner//" which should result in an empty repo name if parsed that way by regex.
        # However, the regex `([^\/]+?)` for repo name requires at least one character.
        with self.assertRaisesRegex(ValueError, invalid_message_regex):
            parse_github_url("http://github.com/owner//")
        with self.assertRaisesRegex(ValueError, invalid_message_regex):
            parse_github_url("https://github.com/user/repo/tree/main") # Should only parse base repo URL

    def test_parse_github_url_different_schemes(self):
        # Updated to reflect that .git is now always removed
        self.assertEqual(parse_github_url("git@github.com:owner/repo.git"), ("owner", "repo")) # SSH URL
        self.assertEqual(parse_github_url("https://github.com/owner/another.repo.git"), ("owner", "another.repo")) # HTTPS with .git
        # Note: The current parse_github_url is simple and might not robustly handle all git URL formats.
        # For SSH, it relies on the split by '/' and ':', which works for this common case.
        # A more robust parser would be needed for all edge cases of git URLs.

if __name__ == "__main__":
    unittest.main()
