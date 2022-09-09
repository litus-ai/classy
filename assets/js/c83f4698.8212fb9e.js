"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[4388],{3905:(e,t,a)=>{a.d(t,{Zo:()=>o,kt:()=>m});var r=a(7294);function s(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function i(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,r)}return a}function n(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?i(Object(a),!0).forEach((function(t){s(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):i(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function c(e,t){if(null==e)return{};var a,r,s=function(e,t){if(null==e)return{};var a,r,s={},i=Object.keys(e);for(r=0;r<i.length;r++)a=i[r],t.indexOf(a)>=0||(s[a]=e[a]);return s}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)a=i[r],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(s[a]=e[a])}return s}var l=r.createContext({}),p=function(e){var t=r.useContext(l),a=t;return e&&(a="function"==typeof e?e(t):n(n({},t),e)),a},o=function(e){var t=p(e.components);return r.createElement(l.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},u=r.forwardRef((function(e,t){var a=e.components,s=e.mdxType,i=e.originalType,l=e.parentName,o=c(e,["components","mdxType","originalType","parentName"]),u=p(a),m=s,f=u["".concat(l,".").concat(m)]||u[m]||d[m]||i;return a?r.createElement(f,n(n({ref:t},o),{},{components:a})):r.createElement(f,n({ref:t},o))}));function m(e,t){var a=arguments,s=t&&t.mdxType;if("string"==typeof e||s){var i=a.length,n=new Array(i);n[0]=u;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c.mdxType="string"==typeof e?e:s,n[1]=c;for(var p=2;p<i;p++)n[p]=a[p];return r.createElement.apply(null,n)}return r.createElement.apply(null,a)}u.displayName="MDXCreateElement"},4906:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>l,contentTitle:()=>n,default:()=>d,frontMatter:()=>i,metadata:()=>c,toc:()=>p});var r=a(7462),s=(a(7294),a(3905));const i={title:"classy.scripts.cli.demo",toc_min_heading_level:2,toc_max_heading_level:4,pagination_next:null,pagination_prev:null},n=void 0,c={unversionedId:"api/scripts/cli/demo",id:"api/scripts/cli/demo",title:"classy.scripts.cli.demo",description:"Functions",source:"@site/docs/api/scripts/cli/demo.md",sourceDirName:"api/scripts/cli",slug:"/api/scripts/cli/demo",permalink:"/classy/docs/api/scripts/cli/demo",draft:!1,tags:[],version:"current",frontMatter:{title:"classy.scripts.cli.demo",toc_min_heading_level:2,toc_max_heading_level:4,pagination_next:null,pagination_prev:null},sidebar:"apiSidebar"},l={},p=[{value:"Functions",id:"functions",level:2},{value:"get_parser",id:"get_parser",level:3},{value:"parse_args",id:"parse_args",level:3},{value:"populate_parser",id:"populate_parser",level:3}],o={toc:p};function d(e){let{components:t,...a}=e;return(0,s.kt)("wrapper",(0,r.Z)({},o,a,{components:t,mdxType:"MDXLayout"}),(0,s.kt)("h2",{id:"functions"},"Functions"),(0,s.kt)("div",{className:"api"},(0,s.kt)("h3",{id:"get_parser"},"get_parser"),(0,s.kt)("div",{className:"api__signature"},"def ",(0,s.kt)("span",{className:"ident"},"get_parser"),"(",(0,s.kt)("br",null),"\xa0\xa0\xa0\xa0subparser=None,",(0,s.kt)("br",null),") \u2011>\xa0argparse.ArgumentParser",(0,s.kt)("div",{className:"links-div"},(0,s.kt)("a",{href:"#get_parser",className:"direct-link"},"#"),(0,s.kt)("a",{href:"https://github.com/sunglasses-ai/classy/blob/2ade6cc06ca6ed0708984f5c3ab3ddb7a2d57759/classy/scripts/cli/demo.py#L36-L50",className:"git-link"},"#"))),(0,s.kt)("div",{className:"api__body"},(0,s.kt)("div",{className:"api__description"}))),(0,s.kt)("div",{className:"api"},(0,s.kt)("h3",{id:"parse_args"},"parse_args"),(0,s.kt)("div",{className:"api__signature"},"def ",(0,s.kt)("span",{className:"ident"},"parse_args"),"()",(0,s.kt)("div",{className:"links-div"},(0,s.kt)("a",{href:"#parse_args",className:"direct-link"},"#"),(0,s.kt)("a",{href:"https://github.com/sunglasses-ai/classy/blob/2ade6cc06ca6ed0708984f5c3ab3ddb7a2d57759/classy/scripts/cli/demo.py#L53-L54",className:"git-link"},"#"))),(0,s.kt)("div",{className:"api__body"},(0,s.kt)("div",{className:"api__description"}))),(0,s.kt)("div",{className:"api"},(0,s.kt)("h3",{id:"populate_parser"},"populate_parser"),(0,s.kt)("div",{className:"api__signature"},"def ",(0,s.kt)("span",{className:"ident"},"populate_parser"),"(",(0,s.kt)("br",null),"\xa0\xa0\xa0\xa0parser:\xa0argparse.ArgumentParser,",(0,s.kt)("br",null),")",(0,s.kt)("div",{className:"links-div"},(0,s.kt)("a",{href:"#populate_parser",className:"direct-link"},"#"),(0,s.kt)("a",{href:"https://github.com/sunglasses-ai/classy/blob/2ade6cc06ca6ed0708984f5c3ab3ddb7a2d57759/classy/scripts/cli/demo.py#L12-L33",className:"git-link"},"#"))),(0,s.kt)("div",{className:"api__body"},(0,s.kt)("div",{className:"api__description"}))))}d.isMDXComponent=!0}}]);