# 博客文章同步仓库

这个仓库用于在不同电脑之间同步博客文章，方便在其他设备上获取 Markdown 文件并完成博客发布。

## 在另一台电脑上使用

首次使用时，克隆仓库：

```bash
git clone https://github.com/HP-Patience/blog_essay.git
cd blog_essay
```

以后每次发布博客前，先拉取最新文章：

```bash
git pull
```

完成编辑或发布准备后，将修改同步回仓库：

```bash
git status
git add "文章文件.md"
git commit -m "更新博客文章"
git push
```

## 目录说明

- 仓库根目录：当前使用的博客文章
- `Archive/`：文章草稿或历史版本
- `.codegraph/`：本地 CodeGraph 配置

为避免不同设备上的修改互相覆盖，每次开始编辑前先执行 `git pull`，完成后及时提交并推送。
