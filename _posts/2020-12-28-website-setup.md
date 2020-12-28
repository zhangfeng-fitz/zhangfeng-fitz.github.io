---
categories:
  - 个人网站
tags:
  - 'GitHub Pages'
date: '2020-12-28'
slug: website-setup
title: GitHub搭建个人网站
---

个人网站（技术博客）对于IT从业者的重要性不言而喻。搭建个人网站可以选择自己租赁服务器进行搭建，不过这种方式对于技术水平有一定要求，且较为繁琐，容易将精力分散在网站的建设运营上。

GitHub是全球最大的开源代码托管网站。截止到2015年，GitHub已经有超过2800万注册用户和5700万代码库，事实上已经成为了世界上最大的代码存放网站和开源社区。GitHub版本控制十分便捷，对个人用户完全免费，且无需关注运营问题，可以专注于博文内容。因此，考虑将网站部署在GitHub上，借助GitHub进行网站管理。

***

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

**GitHub**官方推荐的编辑工具为**jekyll**，因此采用jekyll进行编辑。



