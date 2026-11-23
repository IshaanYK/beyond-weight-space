import subprocess
import sys

def run_cmd(cmd):
    print(f"[*] Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)
    return res.returncode

def main():
    print("=" * 60)
    print("[*] DEPLOYING TO GITHUB PAGES & UPDATING MAIN")
    print("=" * 60)
    
    # 1. Commit any remaining changes on main
    run_cmd('git add -A')
    run_cmd('git commit -m "feat(site): update to Jiajun Fan editorial theme with light/dark mode"')
    
    # 2. Push main
    print("\n[*] Pushing to origin main...")
    rc1 = run_cmd('git push origin main')
    
    # 3. Push to gh-pages branch
    print("\n[*] Pushing to origin gh-pages...")
    rc2 = run_cmd('git push origin main:gh-pages --force')
    
    if rc1 == 0 and rc2 == 0:
        print("\n🎉 SITE SUCCESSFULLY DEPLOYED TO GITHUB PAGES!")
        print("Live URL: https://ishaanyk.github.io/beyond-weight-space/")
    else:
        print("\n[-] Push failed. Check credentials or remote.")

if __name__ == "__main__":
    main()
