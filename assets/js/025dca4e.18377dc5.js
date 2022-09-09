"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[1675],{3905:(e,t,r)=>{r.d(t,{Zo:()=>p,kt:()=>f});var n=r(7294);function i(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function o(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function a(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?o(Object(r),!0).forEach((function(t){i(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function s(e,t){if(null==e)return{};var r,n,i=function(e,t){if(null==e)return{};var r,n,i={},o=Object.keys(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||(i[r]=e[r]);return i}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(i[r]=e[r])}return i}var c=n.createContext({}),l=function(e){var t=n.useContext(c),r=t;return e&&(r="function"==typeof e?e(t):a(a({},t),e)),r},p=function(e){var t=l(e.components);return n.createElement(c.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},u=n.forwardRef((function(e,t){var r=e.components,i=e.mdxType,o=e.originalType,c=e.parentName,p=s(e,["components","mdxType","originalType","parentName"]),u=l(r),f=i,m=u["".concat(c,".").concat(f)]||u[f]||d[f]||o;return r?n.createElement(m,a(a({ref:t},p),{},{components:r})):n.createElement(m,a({ref:t},p))}));function f(e,t){var r=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var o=r.length,a=new Array(o);a[0]=u;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:i,a[1]=s;for(var l=2;l<o;l++)a[l]=r[l];return n.createElement.apply(null,a)}return n.createElement.apply(null,r)}u.displayName="MDXCreateElement"},8870:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>c,contentTitle:()=>a,default:()=>d,frontMatter:()=>o,metadata:()=>s,toc:()=>l});var n=r(7462),i=(r(7294),r(3905));const o={sidebar_position:3,title:"Choosing a profile"},a=void 0,s={unversionedId:"getting-started/basic/choosing-profile",id:"getting-started/basic/choosing-profile",title:"Choosing a profile",description:"This step is not mandatory, but we highly recommend you to read it as it touches an important component of",source:"@site/docs/getting-started/basic/choosing-profile.md",sourceDirName:"getting-started/basic",slug:"/getting-started/basic/choosing-profile",permalink:"/classy/docs/getting-started/basic/choosing-profile",draft:!1,editUrl:"https://github.com/sunglasses-ai/classy/edit/main/docs/docs/getting-started/basic/choosing-profile.md",tags:[],version:"current",sidebarPosition:3,frontMatter:{sidebar_position:3,title:"Choosing a profile"},sidebar:"tutorialSidebar",previous:{title:"Organizing your data",permalink:"/classy/docs/getting-started/basic/data-formatting"},next:{title:"Training your model",permalink:"/classy/docs/getting-started/basic/train"}},c={},l=[],p={toc:l};function d(e){let{components:t,...r}=e;return(0,i.kt)("wrapper",(0,n.Z)({},p,r,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("admonition",{type:"tip"},(0,i.kt)("p",{parentName:"admonition"},"This step is ",(0,i.kt)("strong",{parentName:"p"},"not mandatory"),", but we highly recommend you to read it as it touches an important component of\n",(0,i.kt)("inlineCode",{parentName:"p"},"classy"),", the profiles, which is needed in case you want to heavily modify your training configuration.")),(0,i.kt)("p",null,"It might be the case that you have constraints of any sort (hardware, performance-wise, etc.), and you might\nbe interested in knowing how to change the default underlying model / optimizer used to train in order to either\nfit in smaller GPUs, be faster, or achieve higher accuracy."),(0,i.kt)("p",null,"In ",(0,i.kt)("inlineCode",{parentName:"p"},"classy"),", this is achieved through ",(0,i.kt)("em",{parentName:"p"},"Profiles"),", which a user can employ as a way of changing the training configuration\nof their model to fit different criteria."),(0,i.kt)("p",null,(0,i.kt)("inlineCode",{parentName:"p"},"classy")," comes with a predefined set of profiles, which you can find ",(0,i.kt)("a",{parentName:"p",href:"/docs/reference-manual/profiles/"},"here"),".\nThe list includes the underlying transformer model, optimizer and a few key features that each profile shines for."),(0,i.kt)("p",null,"For this tutorial, we'll stick with a fast yet powerful model, ",(0,i.kt)("em",{parentName:"p"},"DistilBERT"),"."))}d.isMDXComponent=!0}}]);