# How to Contribute

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

### Guidelines for Pull Requests

*   Create **small PRs** that are narrowly focused on addressing a single
    concern. We often times receive PRs that are trying to fix several things at
    a time, but only one fix is considered acceptable, nothing gets merged and
    both author's and review's time is wasted. See
    [Small CLs](https://google.github.io/eng-practices/review/developer/small-cls.html)
    for more details.

*   For speculative changes, consider opening an
    [issue](https://github.com/google-deepmind/alphagenome/issues) and
    discussing it first.

*   Provide a good **PR description** as a record of **what** change is being
    made and **why** it was made. Link to a GitHub issue if it exists.

*   Don't fix code style and formatting unless you are already changing that
    line to address an issue. PRs with irrelevant changes won't be merged.

*   Unless your PR is trivial, you should expect there will be reviewer comments
    that you'll need to address before merging. We expect you to be reasonably
    responsive to those comments, otherwise the PR will be closed after 2-3
    weeks of inactivity.

*   Maintain **clean commit history** and use **meaningful commit messages**.
    PRs with messy commit history are difficult to review and won't be merged.
    Use `rebase -i upstream/main` to curate your commit history and/or to bring
    in latest changes from main (but avoid rebasing in the middle of a code
    review).

*   Keep your PR up to date with upstream/main (if there are merge conflicts, we
    can't really merge your change).

*   **All tests need to be passing** and code formatted before your change can
    be merged. We recommend you run tests locally **before** sending a PR (see
    [Contributing](README.md#contributing) for how to check formatting and run
    tests).

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).
