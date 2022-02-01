const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

// With JSDoc @type annotations, IDEs can provide config autocompletion
/** @type {import('@docusaurus/types').DocusaurusConfig} */
(module.exports = {
  title: 'Classy',
  tagline: 'The NLP toolkit that does not require you to be a programmer! (Kinda 😅)',
  url: 'https://sunglasses-ai.github.io',
  baseUrl: '/classy/',
  trailingSlash: true,
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: '/img/classy_logo-short_transparent.png',
  organizationName: 'sunglasses-ai', // Usually your GitHub org/user name.
  projectName: 'classy', // Usually your repo name.
  presets: [
    [
      '@docusaurus/preset-classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl: ({docPath}) => docPath.startsWith('api/') ? undefined : `https://github.com/sunglasses-ai/classy/edit/main/docs/${docPath}`,
          // editUrl: 'https://github.com/sunglasses-ai/classy/edit/main/docs/',
          include: [
            "/glossary/*.md",
            "**/*.md"
          ],
        },
        // blog: {
        //   showReadingTime: true,
        //   // Please change this to your repo.
        //   editUrl:
        //     'https://github.com/facebook/docusaurus/edit/main/website/blog/',
        // },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    (
      {
        algolia: {
          // The application ID provided by Algolia
          appId: '02OUF7W4JD',
    
          // Public API key: it is safe to commit it
          apiKey: '2e4d4570c14c4dfe5050a9b90d3cff22',
    
          indexName: 'sunglasses-ai-classy',
    
          // Optional: see doc section below
          contextualSearch: true,
        },
        announcementBar: {
          id: 'announcementBar-2', // Increment on change
          content: 'v0.1.0: Classy is still in its early stages, help us shape its future by <a href="https://github.com/sunglasses-ai/classy#contributions">contributing</a>!'
        },
        navbar: {
          title: 'Classy',
          logo: {
            alt: 'Classy Logo',
            src: '/img/CLASSY.svg',
          },
          items: [
            {
              type: 'doc',
              docId: 'intro',
              position: 'left',
              label: 'Tutorial',
            },
            {
              type: 'doc',
              docId: 'api/main',
              position: 'left',
              label: 'Reference API',
            },
            {
              href: 'https://github.com/sunglasses-ai/classy',
              label: 'Classy@GitHub',
              position: 'left',
            },
            {
              href: 'https://github.com/sunglasses-ai/classy-template',
              label: 'Template@GitHub',
              position: 'left',
            },
            {
              href: 'https://github.com/sunglasses-ai/classy-examples',
              label: 'Examples@GitHub',
              position: 'left',
            },
          ],
        },
        footer: {
          links: [
            {
              title: 'Links',
              items: [
                {
                  label: 'Tutorial',
                  to: '/docs/intro',
                },
                {
                  label: 'Getting Started',
                  to: '/docs/installation',
                }
              ],
            },
            {
              title: 'Community',
              items: [
                {
                  label: 'Stack Overflow',
                  href: 'https://stackoverflow.com/questions/tagged/classy',
                },
              ],
            },
            {
              title: 'More',
              items: [
                // {
                //   label: 'Blog',
                //   to: '/blog',
                // },
                {
                  label: 'GitHub',
                  href: 'https://github.com/sunglasses-ai/classy',
                },
                {
                  label: 'Template',
                  href: 'https://github.com/sunglasses-ai/classy-template',
                },
                {
                  label: 'Examples',
                  href: 'https://github.com/sunglasses-ai/classy-examples',
                },
              ],
            },
          ],
          copyright: `Copyright © ${new Date().getFullYear()} sunglasses.ai.`,
        },
        prism: {
          theme: lightCodeTheme,
          darkTheme: darkCodeTheme,
        },
      }),

  plugins: [
    [
        '@docusaurus-terminology/parser',
        {
          termsDir: './docs/glossary/',
        }
    ]
  ],
});
