# papaer-touching

[![Documentation Status](https://readthedocs.org/projects/paper-touching/badge/?version=latest)](https://paper-touching.readthedocs.io/zh_CN/latest/?badge=latest)

log for paper reading or running/re-implementation/implementation.

view doc on [read-the-docs](https://paper-touching.readthedocs.io/zh_CN/latest/index.html)

## rST 的小技巧

- 列表的换行时，只要保证第二行和第一行文字缩进一致就可以。
  切不可第一行开始在 col 3 （类似 `1. `，点后1个空格），第二行直接按 `Tab` 缩进4个空格，这样第二行开始和第一行文字没有对齐，
  第二行会被渲染成 `blockquote`, 会非常不好看的。
  然后，第二行敲 3 个空格可能麻烦，且不容易整体用 `Tab` 调整格式（选中多行，用 Tab 控制缩进），
  那么一个 trick 就是 ——　让第一行文章从　col 4 开始，也就是类似 `1.  ` （点后2个空格），这样后面的多行直接用 Tab，简单有效。
  
  这是从 stackoverflow 看到的，官方的教程里提到了，奈何看不仔细。
