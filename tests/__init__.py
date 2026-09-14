"""Makes ``tests`` a regular package rather than an implicit namespace package.

With this file present, pytest's default ("prepend") import mode always imports
test modules under their fully qualified name (``tests.test_losses``, not a
bare ``test_losses``), and ``tests.helpers`` resolves the same way on every
Python/pytest version and every machine -- no dependence on what else happens
to be on ``sys.path``.
"""
