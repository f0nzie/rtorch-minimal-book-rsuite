# CHANGES

## 0.0.2.9000 (2019-09-22)
* Remove folders `data` and `datasets` from master project.
* Save datasets inside book project, not outside

## 0.0.2 (2019-09-22)
* Modify structure for the book using PART to separate main sections.
* Book structure is:
  1. Getting started
  2. Basic Tensor Operations
  3. Logistic Regression
  4. Linear Regression
  5. Neural Networks
  6. PyTorch and R data structures
  7. Appendix
* Numbering follows the PART number. Example: 0301-, 0302-, 0303, for consecutive chapters. 
* Rename notebooks consecutive. Restart at new section.
* Folder `work` for temporary books or testing.
* Main book resides in `rtorch-bookdown`.
* Moving notebooks that are completed to folder `advanced`.
* Changed named of repository to `rtorch-minimal-book.git`.


## 0.0.1 (2019-09-10)
* Added labels to all chunks
* All notebooks knit alright
* Use `message()` to indicate what a command does
* Selected `new_session: yes` to force individual notebook without dependencies.
* Created bookdown `isolation` to test failing notebooks
* Using R 3.6
* Remove label `prereqs` from index.Rmd
