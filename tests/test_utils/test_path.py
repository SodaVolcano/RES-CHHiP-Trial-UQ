from typing import Generator
from unittest import mock
from pathlib import Path

from ..context import PATCH_LIST_FILES, PATCH_OS_WALK, utils

generate_full_paths = utils.generate_full_paths
list_files = utils.list_files
resolve_path_placeholders = utils.resolve_path_placeholders
next_available_path = utils.next_available_path


class TestListFiles:

    # returns all files in a directory
    def test_returns_all_files_in_directory(self, mocker):
        mocker.patch(
            PATCH_OS_WALK,
            return_value=[
                ("/path", ("subdir",), ("file1.txt", "file2.txt", "file$pec!al.txt")),
                ("/path/subdir", (), ("file3.txt",)),
            ],
        )

        result = list(list_files("/path"))
        expected = [
            "/path/file1.txt",
            "/path/file2.txt",
            "/path/file$pec!al.txt",
            "/path/subdir/file3.txt",
        ]
        assert result == expected

    # empty directory returns no files
    def test_empty_directory_returns_no_files(self, mocker):
        mocker.patch(PATCH_OS_WALK, return_value=[("/empty_path", (), ())])

        result = list(list_files("/empty_path"))
        assert result == []

    # directory with only subdirectories returns no files
    def test_directory_with_only_subdirectories_returns_no_files(self, mocker):
        mocker.patch(
            PATCH_OS_WALK,
            return_value=[
                ("/path", ("subdir1", "subdir2"), ()),
                ("/path/subdir1", (), ()),
                ("/path/subdir2", (), ()),
            ],
        )

        result = list(list_files("/path"))
        expected = []
        assert list(result) == expected

    # handles directories with hidden files
    def test_handles_directories_with_hidden_files(self, mocker):
        mocker.patch(
            PATCH_OS_WALK,
            return_value=[
                ("/path", ("subdir",), ("file1.txt", ".hidden_file", "file2.txt")),
                ("/path/subdir", (), ("file3.txt",)),
            ],
        )

        result = list(list_files("/path"))
        expected = ["/path/file1.txt", "/path/file2.txt", "/path/subdir/file3.txt"]
        assert result == expected


class TestGenerateFullPaths:

    # generates full paths correctly when root and path_generator provide valid inputs
    def test_generates_full_paths_correctly(self):
        def mock_path_generator(root):
            return ["file1.txt", "file2.txt", "dir/file3.txt"]

        root = "/home/user"
        expected_paths = [
            "/home/user/file1.txt",
            "/home/user/file2.txt",
            "/home/user/dir/file3.txt",
        ]

        result = list(generate_full_paths(root, mock_path_generator))
        assert result == expected_paths

    # handles empty strings for root and paths
    def test_handles_empty_strings(self):
        def mock_path_generator(root):
            return ["", "file.txt", "dir/"]

        root = ""
        expected_paths = ["", "file.txt", "dir/"]

        result = list(generate_full_paths(root, mock_path_generator))
        assert result == expected_paths

    # handles path_generator returning an empty list
    def test_handles_empty_path_generator(self):
        def mock_path_generator(root):
            return []

        root = "/home/user"
        expected_paths = []

        result = list(generate_full_paths(root, mock_path_generator))
        assert result == expected_paths

    # ensures generator is lazy and does not compute paths upfront
    def test_lazy_generation(self):
        def mock_path_generator(root):
            return ["file1.txt", "file2.txt", "dir/file3.txt"]

        root = "/home/user"
        expected_paths = [
            "/home/user/file1.txt",
            "/home/user/file2.txt",
            "/home/user/dir/file3.txt",
        ]

        result = generate_full_paths(root, mock_path_generator)
        assert isinstance(result, Generator)
        assert list(result) == expected_paths


class TestResolvePathPlaceholders:

    # Resolves placeholders correctly for simple path patterns with single-level directories
    def test_resolves_placeholders_simple_path(self, mocker):

        # Mock list_files to return a predefined list of files
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=["/a/val1/val2/d/e.jpg", "/a/val3/val4/d/e.jpg"],
        )

        path_pattern = "/a/{b}/{c}/d/{e}.jpg"
        placeholders = ["b", "c"]

        expected = ["/a/val1/val2/d/{e}.jpg", "/a/val3/val4/d/{e}.jpg"]

        result = resolve_path_placeholders(path_pattern, placeholders)
        assert result == expected

    # Handles empty directory gracefully
    def test_handles_empty_directory(self, mocker):

        # Mock list_files to return an empty list
        mocker.patch(PATCH_LIST_FILES, return_value=[])

        path_pattern = "/a/{b}/{c}/d/{e}.jpg"
        placeholders = ["b", "c"]

        expected = []

        result = resolve_path_placeholders(path_pattern, placeholders)
        assert result == expected

    # Handles non-existent directories
    def test_handles_non_existent_directories(self, mocker):

        # Mock list_files to return an empty list for non-existent directories
        mocker.patch(PATCH_LIST_FILES, return_value=[])

        path_pattern = "/non_existent/{b}/{c}/d/{e}.jpg"
        placeholders = ["b", "c"]

        expected = []

        result = resolve_path_placeholders(path_pattern, placeholders)
        assert result == expected

    # Handles patterns with no placeholders
    def test_handles_patterns_with_no_placeholders(self, mocker):

        # Mock list_files to return a predefined list of files
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=["/a/val1/val2/d/e.jpg", "/a/val3/val4/d/e.jpg"],
        )

        path_pattern = "/a/val1/val2/d/e.jpg"
        placeholders = []

        expected = [path_pattern]

        result = resolve_path_placeholders(path_pattern, placeholders)
        assert result == expected

    # Test that when string have leading empty placeholders, only specified placeholder is replaced
    def test_leading_empty_placeholders_replacement(self, mocker):

        # Mock list_files to return a predefined list of files
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=[
                "/a/val1/val2/d/e.jpg",
                "/a/val3/val4/d/e.jpg",
                "/a/val5/val6/d/e.jpg",
                "/a/val7/val8/d/e.jpg",
            ],
        )

        path_pattern = "/{}/{t1}/{t2}/d/{t3}.jpg"
        placeholders = ["t2"]

        expected = [
            "/{}/{t1}/val2/d/{t3}.jpg",
            "/{}/{t1}/val4/d/{t3}.jpg",
            "/{}/{t1}/val6/d/{t3}.jpg",
            "/{}/{t1}/val8/d/{t3}.jpg",
        ]

        result = resolve_path_placeholders(path_pattern, placeholders)
        assert result == expected

    # Test that replacement is correct using placeholders with diferent order than in path
    def test_different_order_placeholders(self, mocker):

        # Mock list_files to return a predefined list of files
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=[
                "/a/val1/val2/d/e.jpg",
                "/a/val3/val4/d/e.jpg",
                "/a/val5/val6/d/e.jpg",
                "/a/val7/val8/d/e.jpg",
            ],
        )

        path_pattern = "/{}/{t1}/{t2}/d/{t3}.jpg"
        placeholders = ["t3", "t1", "t2"]

        expected = resolve_path_placeholders(path_pattern, ["t1", "t2", "t3"])

        result = resolve_path_placeholders(path_pattern, placeholders)
        assert result == expected


class TestNextAvailablePath:

    # Returns original path when it does not exist
    def test_returns_original_path_when_not_exists(self, mocker):
        test_path = "/test/path/file"
        mocker.patch("os.path.exists", return_value=False)

        result = next_available_path(test_path)

        assert result == Path(test_path)

    # Handles paths with no file extension
    def test_handles_path_without_extension(self, mocker):
        test_path = "/test/path/file"
        exists_mock = mocker.patch("os.path.exists")
        exists_mock.side_effect = [True, False]

        result = next_available_path(test_path)

        assert result == Path("/test/path/file-0")

    # Returns path with incremented suffix when previous suffixed paths exist
    def test_incremented_suffix_for_existing_paths(self, mocker):
        # Mock os.path.exists to simulate existing files
        mocker.patch(
            "os.path.exists",
            side_effect=lambda x: x in {"file.txt", "file-0.txt", "file-1.txt"},
        )

        # Call the function with a path that already exists
        result = next_available_path("file.txt")

        # Assert that the result is the next available path
        assert result == Path("file-2.txt")

    # Handles paths with file extensions correctly
    def test_handles_existing_file_without_extension(self):
        # Mock os.path.exists to simulate existing files
        with mock.patch("os.path.exists") as mock_exists:
            # Simulate that 'file.txt' and 'file-1.txt' exist
            mock_exists.side_effect = lambda path: path in ["file", "file-0"]

            # Call the function with a path that has an extension
            result = next_available_path("file")

            # Assert that the function returns the next available path with the correct extension
            assert result == Path("file-1")
