# Website

This website is built using [Docusaurus 2](https://docusaurus.io/), a modern static website generator.

### Local Development

We suggest using Docker to avoid npm/yarn installations. Simply run:
```
$ bash local-dev.sh
```
and follow the interactive prompt.

### Build

```
$ yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

```
$ GIT_USER=<Your GitHub username> USE_SSH=true yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
