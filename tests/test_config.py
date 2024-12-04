from .context import config

auto_match_config = config.auto_match_config


class TestAutoMatchConfig:

    # Decorator correctly passes matching config values to function parameters
    def test_decorator_passes_matching_config(self):
        @auto_match_config(prefixes=["test"])
        def sample_func(a: int, b: str, c: float, d: str = "default"):
            return a, b, c, d

        config = {
            "test__a": 42,
            "test__b": "hello",
            "test__c": 3.14,
            "test__hello": 42,
        }

        result = sample_func(**config)  # type: ignore
        assert result == (42, "hello", 3.14, "default")

    # Empty config dictionary is handled gracefully
    def test_empty_config_handling_with_prefixes(self):
        @auto_match_config(prefixes=["test"])
        def sample_func(a: int = 1, b: str = "default"):
            return a, b

        config = {}

        result = sample_func(**config)  # type: ignore
        assert result == (1, "default")

    # Empty config dictionary is handled gracefully
    def test_empty_config_handling(self):
        @auto_match_config(prefixes=["1"])
        def sample_func(a, b):
            return a + b

        config = {}

        result = sample_func(5, 3, **config)  # type: ignore
        assert result == 8

    # Function executes successfully without prefixes and direct config dictionary
    def test_auto_match_config_without_prefixes(self):
        @auto_match_config(prefixes=["4"])
        def multiply(x, y):
            return x * y

        config = {"4__x": 3, "4__y": 4}
        result = multiply(**config)
        assert result == 12

    # test that wrapper works for class methods
    def test_wrapper_with_class_method(self):
        class Calculator:
            @auto_match_config(prefixes=["init"])
            def __init__(self, a: int, b: int):
                self.a = a
                self.b = b

            @auto_match_config(prefixes=["operation"])
            def add(self, c: int, b: int):
                return c + b

        config = {"operation__c": 3, "operation__b": 4, "init__a": 1, "init__b": 2}
        calculator = Calculator(**config)  # type: ignore
        result = calculator.add(**config)  # type: ignore
        assert result == 7
        assert calculator.a == 1
        assert calculator.b == 2

    # test that remaining config values are passed to kwargs
    def test_passing_kwargs_to_inner_function(self):
        @auto_match_config(prefixes=["1"])
        def test2(c, d, **kwargs):
            res = test3(**kwargs)
            return res * (c + d)

        @auto_match_config(prefixes=["1"])
        def test3(e, f):
            return e + f

        @auto_match_config(prefixes=["1"])
        def test(a, b, **kwargs):
            res = test2(**kwargs)
            return res * (a - b)

        config = {
            "1__a": 8,
            "1__b": 2,
            "1__c": 30,
            "1__d": 4,
            "1__e": 5,
            "1__f": 6,
            "1__hello!": 7,
            "1__world!": 8,
            "2__never": 1,
            "2__gonna": 2,
            "2__give": 3,
            "2__you": 4,
            "2__up!": 5,
        }

        result = test(**config)
        assert result == 2244

    # test that parameters are correctly merged when specifying multiple prefixeses
    def test_multiple_prefixeses(self):
        config = {
            "test__a": 42,
            "test__hello": 42,
            "hello__b": "hello",
            "hello__world": 43,
            "world__c": 90,
            "world__hello": 44,  # overwrites hello from "test"
        }

        @auto_match_config(prefixes=["test", "world"])
        def test(a, c, hello):
            return a, c, hello

        result = test(**config)
        assert result == (42, 90, 44)

    def test_no_config_passed_in(self):
        @auto_match_config(prefixes=["test"])
        def test(a, b):
            return a, b

        result = test(1, 2)
        assert result == (1, 2)

    # Test that kwargs in config used in outer function are passed to inner function as well
    def test_config_param_passed_to_inner_function_with_same_param(self):
        def test_kwargs_and_config(self):
            config = {"test__a": 1, "test__b": -1234}

            @auto_match_config(prefixes=["test"])
            def test(a, b, c):
                return a, b, c

            result = test(c=4, b=2, **config)  # b=2 overrides config["test__b"]

            assert result == (1, 2, 4)

        @auto_match_config(prefixes=["a"])
        def test(a, b, c, **kwargs):
            b2, e, f = test2(**kwargs)
            return a * b + c * b2 + e * f

        @auto_match_config(prefixes=["a"])  # b is the param used in test() as well
        def test2(b, e, f):
            return b, e, f

        config = {"a__a": 1, "a__b": 2, "a__c": 3, "a__d": 4, "a__e": 5, "a__f": 6}
        result = test(**config)

        assert result == 38
