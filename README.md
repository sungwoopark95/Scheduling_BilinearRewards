# Github Push and Pull
* Push: 로컬 repository에 추가 및 변경된 사항을 remote repository(git)에 업로드하는 작업
* Pull: Remote repository에 추가 및 변경된 사항을 로컬 repository로 불러와 업데이트하는 작업
* Remote repository와 로컬 repository가 서로 동기화되어 있어야 push할 수 있기 때문에, 항상 pull부터 해주어야 함
* Pull & Push하는 방법과 순서
```bash
git pull origin main # pull request
git add .
git commit -m "message" # message에는 기록하고 싶은 내용들 (주로 업데이트 사항)
git push origin main
```
* Git Conflict가 나서 새로 Github 내용으로 덮어쓰고 싶을 때
```bash
git fetch --all
git reset --hard origin/main
git pull origin main
```
