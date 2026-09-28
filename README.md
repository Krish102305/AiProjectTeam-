# AiProjectTeam - Hello World

`hello.py` prints a greeting followed by the list of group members.

```
python hello.py
```

## Adding your name

Each member adds **their own** name through their own pull request:

1. Clone the repo and create a branch:
   ```
   git clone https://github.com/Krish102305/AiProjectTeam-.git
   cd AiProjectTeam-
   git checkout -b add-<yourname>
   ```
2. Add your name to the `members` list in `hello.py`, e.g. `members = ["Krish"]`.
   If other names are already there, add yours after them: `["Krish", "Alex"]`.
3. Run `python hello.py` to make sure it still works.
4. Commit and push:
   ```
   git commit -am "Add <yourname> to group members"
   git push -u origin add-<yourname>
   ```
5. On GitHub, open a pull request into `main` and request a review from a teammate.
6. The teammate reviews and approves (**Files changed → Review changes → Approve**).
7. Merge the PR once it's approved.

If `main` has changed since you branched (someone else merged first), run
`git pull origin main` on your branch, keep **both** names in the list, commit, and push.
