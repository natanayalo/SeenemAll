# Review Resolution Checklist

Use these commands as a practical sequence.

## 1. PR Summary
```bash
gh pr view <PR_NUMBER> --repo <OWNER>/<REPO> --json number,title,state,updatedAt,reviews,comments
```

## 2. Inline Review Comments
```bash
gh api repos/<OWNER>/<REPO>/pulls/<PR_NUMBER>/comments --paginate
```

## 3. Review Threads (Unresolved Focus)
```bash
gh api graphql -f owner='<OWNER>' -f repo='<REPO>' -F number=<PR_NUMBER> -f query='query($owner:String!, $repo:String!, $number:Int!){ repository(owner:$owner,name:$repo){ pullRequest(number:$number){ reviewThreads(first:100){ nodes{ id isResolved isOutdated path comments(first:1){ nodes{ url body author{login} createdAt } } } } } } }'
```

## 4. Validation
```bash
pre-commit run --all-files
pytest
```

For large repos, run targeted tests for touched areas first, then full tests if required by policy.

## 5. Resolve Addressed Threads
```bash
gh api graphql -f query='mutation($id:ID!){ resolveReviewThread(input:{threadId:$id}) { thread { id isResolved } } }' -f id=<THREAD_ID>
```

Resolve only threads that are implemented and pushed.

## 6. Verify Final Thread Status
```bash
gh api graphql -f owner='<OWNER>' -f repo='<REPO>' -F number=<PR_NUMBER> -f query='query($owner:String!, $repo:String!, $number:Int!){ repository(owner:$owner,name:$repo){ pullRequest(number:$number){ reviewThreads(first:100){ nodes{ id isResolved path } } } } }'
```

## 7. Final Report Template
- Implemented threads: <links>
- Deferred threads: <links + rationale>
- Validation results: <commands + pass/fail>
- Final unresolved count: <number>
- Commit hash(es): <sha>
