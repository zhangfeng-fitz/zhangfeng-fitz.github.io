---
categories:
  - 个人网站
tags:
  - ’Next Theme'
date: '2020-12-29'
slug: website-next
title: 使用Next主题配置网站
---



GitHub Pages的默认主题不太美观，且需要自己编写代码实现前端相关功能，较为繁琐。Next主题为黑白配色，风格极简，是个人非常喜欢的一款主题。本文介绍如何使用Next主题进行个人网站配置。

<!-- more-->

***

本站采用的是GitHub的默认编辑框架Jekyll进行编辑，因此需要下载Jekyll支持的[Next主题](https://github.com/Simpleyyt/jekyll-theme-next)。

下载之后进行将压缩包中的文件与个人网站的同名文件进行对应替换，`_post`文件夹下的文件不需要替换，因为`_post`文件夹下存放的是博客文章。

下载替换完成后，最关键的是更改`Next`配置以实现启用不同功能,`Next`的相关配置放在`_config.yml`文件中，因此对该文件进行修改即可，具体可参阅[Next使用文档](https://theme-next.iissnan.com/)。接下来介绍本网站进行的相关配置。

# 界面设定

## 设定外观

`Next`默认支持的主题有三种，分别是`Muse`（默认版本，黑白主调，大量留白）、`Mist`（紧凑版本，单栏外观）、`Pisces`（双栏外观）。在配置文件中，三种外观以注释的形式出现，只需要选择喜欢的外观去掉注释即可启用。

## 设置菜单

菜单设置可以设置菜单项。菜单项对应的代码如下，去掉对应的注释即可启用对应的菜单。

```yaml
menu:
  home: /
  #archives: /archives
  #about: /about
  #categories: /categories
  #tags: /tags
  #commonweal: /404.html
```

此外，还可以修改菜单名称、菜单图标等，本站未作修改，不做赘述。

## 设置站点概览

可以设置本站的基本情况概览，包括站名、作者、描述等信息。对应代码为

```yaml
# Site
title: Fitz
subtitle: Zhang Feng's Blog
description: Keep Curious
author: Fitz
```

## 设置社交链接

`social`字段可用于设置社交链接，以及对应的图标。不需要某行可将其注释掉，需要可直接更改地址即可。若不需要图标，则将`social_icons`中`enable`的值更改为`false`。

```yaml
social:
  GitHub: https://github.com/zhangfeng-fitz
  CSDN: https://blog.csdn.net/qq_29798939?spm=1001.2014.3001.5343
  Email: fengzhang2018@163.com
  LeetCode: https://leetcode-cn.com/u/zhangfeng-fitz/
social_icons:
  enable: true
  GitHub: github
  Twitter: twitter
  Weibo: weibo
  CSDN: csdn
  LeetCode: leetcode
```

## 设置建立时间

可以设置显示在网站最下方的网站建立时间。

```yaml
since: 2013
```

## 设置动画效果

```yaml 
use_motion: true  # 开启动画效果
use_motion: false # 关闭动画效果
```

还可以设置背景动画，本站启用的效果为`canvas_nest`

```yaml
# canvas_nest
canvas_nest: true //开启动画
canvas_nest: false //关闭动画
```

# 设置第三方服务

Next可以借助第三方服务来扩展其功能，包括评论、访问量统计、内容分享、站内搜索等功能，本站开启评论、访问量统计，站内搜索等功能，其中站内搜索功能在菜单项中开启搜索菜单后默认开启，下面说明评论和访问量统计功能的开启。

## 访问量统计功能

访问量统计功能主要通过[不蒜子](https://busuanzi.ibruce.info/)实现，Next已内置对于不蒜子的支持，因此可以直接通过更改配置文件实现。将`busuanzi_count`中的`enable`值改为`true`即可，`header`和`footer`字段可以用来设置显示格式。

```yaml 
busuanzi_count:
  # count values only if the other configs are false
  enable: true
  # custom uv span for the whole site
  site_uv: true
  site_uv_header: <i class="fa fa-user"></i> 访问人数
  site_uv_footer: 人
  # custom pv span for the whole site
  site_pv: true
  site_pv_header: <i class="fa fa-eye"></i> 总访问量
  site_pv_footer: 次
  # custom pv span for one page only
  page_pv: true
  page_pv_header: <i class="fa fa-file-o"></i> 阅读数
  page_pv_footer:
```

## 评论功能

评论功能通过`Gitalk`实现。`Gitalk`通过`Github`登录，支持Markdown语法。通过为每一篇博文创建一个`GitHub Issue`来进行评论。启用`Gitalk`主要有两步，首先需要`Github`授权，然后在配置文件中进行配置即可。

### 注册使用GitHub API

在`Github`网页右上角打开`Settings`，找到`Developer Settings`中的`OAuth Apps`，点击`New OAuth App`新建，名称随意写，`Homepage URL`和`Authorization callback URL`都写网站的域名（包含https://）。

![](/img/20201229/oauth.png)

填写完毕后点击注册得到新的OAuth App。

![](/img/20201229/gitalk.png)

### 修改相关配置

复制得到的`ClientID`和`Client Secret`到配置文件中，并修改`repo、owner、admin`等字段即可。

```yaml
gitalk:
  enable: true
  clientID: # 54730ddf7b9420460c03 (your clientID)
  clientSecret: # c05684701b25ff1c19ea0c01ce4d80c125e0707d (your clientSecret)
  repo: zhangfeng-fitz.github.io # colingpt.github.io
  owner: zhangfeng-fitz # colingpt
  admin: zhangfeng-fitz # colingpt
```

至此，相关设置全部完成，后续如果有配置增加再来修改。经过配置，网站已经基本完成，剩下的便是用博文充实网站。希望本站能吸引到越来越多的游客！