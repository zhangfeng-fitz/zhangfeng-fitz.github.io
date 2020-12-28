---
categories:
  - 个人网站
tags:
  - 'GitHub Pages'
date: '2020-12-28'
slug: website-setup
title: GitHub搭建个人网站
---

个人网站（技术博客）对于IT从业者的重要性不言而喻。搭建个人网站可以选择自己租赁服务器进行搭建，不过这种方式对于技术水平有一定要求，且较为繁琐，容易将精力分散在网站的建设运营上。GitHub版本控制十分便捷，对个人用户完全免费，且无需关注运营问题，可以专注于博文内容。因此，考虑将网站部署在GitHub上，借助GitHub进行网站管理。本文主要介绍了如何在GitHub上搭建个人网站。

<!-- more -->

使用GitHub搭建网站可以分为以下几步：

（1）注册GitHub账号

（2）建立个人网站代码仓库

（3）更换网站主题（可选）

（4）安装GitHub Pages编辑工具

（5）为网站自定义域名（可选）

***

# 注册GitHub账号

​		 此步骤较为简单，登陆[Github](https://github.com/)网站，点击右上角的**Sign Up**，编辑用户名、邮箱地址和密码即可注册。注册后登陆邮箱进行验证即完成注册。注：用户名不易更改，谨慎起名。

![](/img/20201228/signup.jpg)

# 建立个人网站代码仓库

​		注册并登录后，点击**Create a Repository**，并在**Repository name**处填入网站的域名，格式为`username.github.io`，将`username`更改为自己的用户名（只能是注册账号时的用户名），仓库类型选择Public，即可完成创建。

![](/img/20201228/repository.jpg)

​		在浏览器中输入**Repository name**即可登录。此时网站为默认状态，非常简陋，接下来为网站更改主题。

# 更换网站主题（可选）

在**Repository**中选择**Settings**，下拉找到**GitHub Pages**，点击**Choose a theme**。

![](/img/20201228/settings.jpg)

选定主题后点击**Select theme**即可应用主题。因为本网站未使用此类主题，所以只做简略介绍。

![](/img/20201228/theme.png)

# 安装GitHub Pages编辑工具

**GitHub**官方推荐的编辑工具为**Jekyll**，因此采用Jekyll进行编辑，以Mac为例。

**Jekyll**需要在**Ruby**环境中使用，因此首先安装**Ruby**环境。

```shell
brew install ruby
```

注：**Mac**中自带**Ruby**，不过版本可能较低，不满足使用需求（需要2.4版本及以上）。可以将Mac的**Ruby**升级至高版本。

安装完成后安装Jekyll。

```shell 
sudo gem install jekyll bundler
```

安装完成之后即完成编辑工具的安装。在编写博客内容时需要使用Markdown语言编写。此处推荐[Typora](https://typora.io/)，此编辑器为极简风格，没有双栏预览设置。将Markdown语法输入完成之后会自动转换为预览效果。个人非常喜欢，强烈推荐！

# 为网站自定义域名

经过上述步骤之后，网站的域名为`username.github.io`，此域名可以直接访问。可以在本网站**Repository**的**Settings**中更改**Custom domain**来进行自定义域名。不过自定义域名存在问题是`.com`类域名无法免费使用，需要首先付费注册。本文将域名自定义为`.com`域名，具体步骤如下。

在[阿里云](https://www.aliyun.com/)中搜索“域名”，购买自己喜欢的域名，此处过程略。购买完成后，将**Custom domain**中的自定义域名改为刚购买的域名并保存。不过此时购买的域名没有对应的DNS解析，因此仍然无法使用，因此需要手动配置DNS。在阿里云中打开**DNS**控制台，在域名解析中添加主域名。

![](/img/20201228/domain.png)

为主域名添加五条DNS记录，对应的记录内容如图所示，将第一条中的记录值改为购买的域名即可。

![](/img/20201228/dns.png)

添加完成后，等待一段时间即可使用域名访问网站。测试可以访问后，在**Repository**的**Settings**中点击**Enforce HTTPS**以开启HTTPS保证安全性（此设置需要在保存**Custom domain**一段时间后才可以进行设置）。

![](/img/20201228/settings.jpg)

最终定义域名的目的为可以通过**GitHub**域名，自定义域名等进行访问。以本站为例，可以通过<https://zhangfeng-fitz.github.io>、<https://zhangfeng-fitz.com>、<https://www.zhangfeng-fitz.com>等来进行访问。

***

至此，个人网站就算搭建完成。不过界面还是略显简陋，后面再对界面进行更改。