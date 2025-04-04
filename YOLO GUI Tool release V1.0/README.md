---
title: 第一篇
date: 2025-03-24 15:18:01
categories: 旅游
tags: 
  - 日本
  - Japan
index_img: /img/first/IMG_0309.JPG
banner_img: /img/first/IMG_0309.JPG
---
<h2>这一页主要测试code</h2>
{% label primary @试试看效果 %}

这里是一张文章内的图片
![枫叶🍁](/img/first/IMG_0309.JPG)
<br></br>
```bash
npm install hexo-clipboard --save
```
<br></br>
{% note success %}
***TIP***
这个算是一个note
{% endnote %}

[**奈良旅游官网**](https://narashikanko.or.jp/cn/guide/ "奈良观光协会")
<br></br>
{% note warning %}
***WARNNING***
这个算是一个Warning
{% endnote %}
{% carousel %}
/img/fnos/4.png | 第一张图片的描述
/img/fnos/5.png | 第二张图片的描述
/img/fnos/6.png | 第三张图片的描述
/img/fnos/7.png | 第四张图片的描述
/img/fnos/8.png | 第五张图片的描述
/img/fnos/9.png | 第六张图片的描述
{% endcarousel %}

{% gi 4 2-2 %}
![奈良JR车站](/img/nara_guide/2.png)
![奈良特色井盖](/img/nara_guide/3.png)
![奈良小鹿](/img/nara_guide/4.png)
![东大寺](/img/nara_guide/5.png)
{% endgi %}

这里是一个video
<!-- {% video /videos/IMG_1873.mp4 %} -->
<!-- 基础用法 -->
<video controls width="100%">
  <source src="/video/IMG_1873.mp4" type="video/mp4">
  您的浏览器不支持视频播放
</video>
