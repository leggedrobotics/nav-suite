# Contribution Guidelines

Nav Suite is a community maintained project. We welcome contributions to the project to make
the framework more mature, extend features and make it useful for everyone.
These may happen in forms of bug reports, feature requests, design proposals and more.

* **Bug reports:** Please report any bugs you find in the [issue tracker](https://github.com/leggedrobotics/nav-suite/issues).
* **Feature requests:** Please suggest new features you would like to see in the [discussions](https://github.com/leggedrobotics/nav-suite/discussions).
* **Code contributions:** Please submit a [pull request](https://github.com/leggedrobotics/nav-suite/pulls) for bug fixes, new features, documentations or tutorials
* **Questions / Ideas:** We prefer GitHub [discussions](https://github.com/leggedrobotics/nav-suite/discussions) for discussing ideas, asking questions, conversations and requests for new features.

Please use the [issue tracker](https://github.com/leggedrobotics/nav-suite/issues) only to track executable pieces of work
with a definite scope and a clear deliverable. These can be fixing bugs, new features, or general updates.

## Contributing Code

> **Note:** Please refer to the [Google Style Guide](https://google.github.io/styleguide/pyguide.html) for the coding style before contributing to the codebase. In the coding style section, we outline the specific deviations from the style guide that we follow in the codebase.

We use [GitHub](https://github.com/leggedrobotics/nav-suite) for code hosting. Please follow the following steps to contribute code:

1. Create an issue in the [issue tracker](https://github.com/leggedrobotics/nav-suite/issues) to discuss the changes or additions you would like to make. This helps us to avoid duplicate work and to make sure that the changes are aligned with the roadmap of the project.
2. Fork the repository.
3. Create a new branch for your changes.
4. Make your changes and commit them.
5. Push your changes to your fork.
6. Submit a pull request to the [main branch](https://github.com/leggedrobotics/nav-suite/compare).
7. Ensure all the checks on the pull request template are performed.

After sending a pull request, the maintainers will review your code and provide feedback.

Please ensure that your code is well-formatted, documented and passes all the tests.

> **Tip:** It is important to keep the pull request as small as possible. This makes it easier for the maintainers to review your code. If you are making multiple changes, please send multiple pull requests. Large pull requests are difficult to review and may take a long time to merge.

## Coding Practices

We adopt the coding practices from IsaacLab. This means in details:

- we maintain a changelog and extension.toml, details [here](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html#maintaining-a-changelog-and-extension-toml)
- we follow the Google Style Guide, as outline in detail [here](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html#coding-style)
- we use pytest for testing
- we use automatic code formatting:

    * [pre-commit](https://pre-commit.com/): Runs a list of formatters and linters over the codebase.
    * [black](https://black.readthedocs.io/en/stable/): The uncompromising code formatter.
    * [flake8](https://flake8.pycqa.org/en/latest/): A wrapper around PyFlakes, pycodestyle and McCabe complexity checker.

    Please check [here](https://pre-commit.com/#install) for instructions to set these up. To run over the entire repository, please execute the following command in the terminal:

    ```bash
    pre-commit run --all-files
    ```

---

## Developer Certificate of Origin
**Version 1.1**

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

### Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
