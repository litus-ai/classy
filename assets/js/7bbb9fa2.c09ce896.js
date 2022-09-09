"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[9087],{3905:(e,t,a)=>{a.d(t,{Zo:()=>d,kt:()=>u});var i=a(7294);function n(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function r(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);t&&(i=i.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,i)}return a}function s(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?r(Object(a),!0).forEach((function(t){n(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):r(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function l(e,t){if(null==e)return{};var a,i,n=function(e,t){if(null==e)return{};var a,i,n={},r=Object.keys(e);for(i=0;i<r.length;i++)a=r[i],t.indexOf(a)>=0||(n[a]=e[a]);return n}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(i=0;i<r.length;i++)a=r[i],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(n[a]=e[a])}return n}var o=i.createContext({}),c=function(e){var t=i.useContext(o),a=t;return e&&(a="function"==typeof e?e(t):s(s({},t),e)),a},d=function(e){var t=c(e.components);return i.createElement(o.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return i.createElement(i.Fragment,{},t)}},m=i.forwardRef((function(e,t){var a=e.components,n=e.mdxType,r=e.originalType,o=e.parentName,d=l(e,["components","mdxType","originalType","parentName"]),m=c(a),u=n,k=m["".concat(o,".").concat(u)]||m[u]||p[u]||r;return a?i.createElement(k,s(s({ref:t},d),{},{components:a})):i.createElement(k,s({ref:t},d))}));function u(e,t){var a=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var r=a.length,s=new Array(r);s[0]=m;var l={};for(var o in t)hasOwnProperty.call(t,o)&&(l[o]=t[o]);l.originalType=e,l.mdxType="string"==typeof e?e:n,s[1]=l;for(var c=2;c<r;c++)s[c]=a[c];return i.createElement.apply(null,s)}return i.createElement.apply(null,a)}m.displayName="MDXCreateElement"},7360:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>o,contentTitle:()=>s,default:()=>p,frontMatter:()=>r,metadata:()=>l,toc:()=>c});var i=a(7462),n=(a(7294),a(3905));const r={title:"classy.optim.optimizers.radam",toc_min_heading_level:2,toc_max_heading_level:4,pagination_next:null,pagination_prev:null},s=void 0,l={unversionedId:"api/optim/optimizers/radam",id:"api/optim/optimizers/radam",title:"classy.optim.optimizers.radam",description:"Classes",source:"@site/docs/api/optim/optimizers/radam.md",sourceDirName:"api/optim/optimizers",slug:"/api/optim/optimizers/radam",permalink:"/classy/docs/api/optim/optimizers/radam",draft:!1,tags:[],version:"current",frontMatter:{title:"classy.optim.optimizers.radam",toc_min_heading_level:2,toc_max_heading_level:4,pagination_next:null,pagination_prev:null},sidebar:"apiSidebar"},o={},c=[{value:"Classes",id:"clzs",level:2},{value:"RAdam",id:"RAdam",level:3},{value:"__init__",id:"RAdam-init",level:4},{value:"step",id:"RAdam-step",level:4}],d={toc:c};function p(e){let{components:t,...a}=e;return(0,n.kt)("wrapper",(0,i.Z)({},d,a,{components:t,mdxType:"MDXLayout"}),(0,n.kt)("h2",{id:"clzs"},"Classes"),(0,n.kt)("div",{className:"api"},(0,n.kt)("h3",{id:"RAdam"},"RAdam"),(0,n.kt)("div",{className:"api__signature"},(0,n.kt)("p",null,"class ",(0,n.kt)("span",{className:"ident"},"RAdam"),"(torch.optim.optimizer.Optimizer)"),(0,n.kt)("div",{className:"links-div"},(0,n.kt)("a",{href:"#RAdam",className:"direct-link"},"#"),(0,n.kt)("a",{href:"https://github.com/sunglasses-ai/classy/blob/2ade6cc06ca6ed0708984f5c3ab3ddb7a2d57759/classy/optim/optimizers/radam.py#L10-L130",className:"git-link"},"#"))),(0,n.kt)("div",{className:"api__body"},(0,n.kt)("div",{className:"api__description"},(0,n.kt)("p",null,"Base class for all optimizers."),(0,n.kt)("div",{class:"admonition warning"},(0,n.kt)("p",{class:"admonition-title"},"Warning"),(0,n.kt)("p",null,"Parameters need to be specified as collections that have a deterministic ordering that is consistent between runs. Examples of objects that don't satisfy those properties are sets and iterators over values of dictionaries.")),(0,n.kt)("h5",{id:"args"},"Args"),(0,n.kt)("dl",null,(0,n.kt)("dt",null,(0,n.kt)("code",null,"params")," :\u2002",(0,n.kt)("code",null,"iterable")),(0,n.kt)("dd",null,"an iterable of :class:",(0,n.kt)("code",null,"torch.Tensor")," s or :class:",(0,n.kt)("code",null,"dict")," s. Specifies what Tensors should be optimized."),(0,n.kt)("dt",null,(0,n.kt)("code",null,"defaults")),(0,n.kt)("dd",null,"(dict): a dict containing default values of optimization options (used when a parameter group doesn't specify them)."))),(0,n.kt)("h4",{id:"RAdam-init"},"_","_","init","_","_"),(0,n.kt)("div",{className:"api__signature"},"def ",(0,n.kt)("span",{className:"ident"},"__init__"),"(",(0,n.kt)("br",null),"\xa0\xa0\xa0\xa0params,",(0,n.kt)("br",null),"\xa0\xa0\xa0\xa0lr=0.001,",(0,n.kt)("br",null),"\xa0\xa0\xa0\xa0betas=(0.9, 0.999),",(0,n.kt)("br",null),"\xa0\xa0\xa0\xa0eps=1e-08,",(0,n.kt)("br",null),"\xa0\xa0\xa0\xa0weight_decay=0,",(0,n.kt)("br",null),"\xa0\xa0\xa0\xa0degenerated_to_sgd=True,",(0,n.kt)("br",null),")",(0,n.kt)("div",{className:"links-div"},(0,n.kt)("a",{href:"#RAdam-init",className:"direct-link"},"#"),(0,n.kt)("a",{href:"https://github.com/sunglasses-ai/classy/blob/2ade6cc06ca6ed0708984f5c3ab3ddb7a2d57759/classy/optim/optimizers/radam.py#L10-L130",className:"git-link"},"#"))),(0,n.kt)("div",{className:"api__description"}),(0,n.kt)("div",{className:"api"},(0,n.kt)("h4",{id:"RAdam-step"},"step"),(0,n.kt)("div",{className:"api__signature"},"def ",(0,n.kt)("span",{className:"ident"},"step"),"(",(0,n.kt)("br",null),"\xa0\xa0\xa0\xa0self,",(0,n.kt)("br",null),"\xa0\xa0\xa0\xa0closure=None,",(0,n.kt)("br",null),")",(0,n.kt)("div",{className:"links-div"},(0,n.kt)("a",{href:"#RAdam-step",className:"direct-link"},"#"),(0,n.kt)("a",{href:"https://github.com/sunglasses-ai/classy/blob/2ade6cc06ca6ed0708984f5c3ab3ddb7a2d57759/classy/optim/optimizers/radam.py#L52-L130",className:"git-link"},"#"))),(0,n.kt)("div",{className:"api__body"},(0,n.kt)("div",{className:"api__description"},(0,n.kt)("p",null,"Performs a single optimization step (parameter update)."),(0,n.kt)("h5",{id:"args"},"Args"),(0,n.kt)("dl",null,(0,n.kt)("dt",null,(0,n.kt)("code",null,"closure")," :\u2002",(0,n.kt)("code",null,"callable")),(0,n.kt)("dd",null,"A closure that reevaluates the model and returns the loss. Optional for most optimizers.")),(0,n.kt)("div",{class:"admonition note"},(0,n.kt)("p",{class:"admonition-title"},"Note"),(0,n.kt)("p",null,"Unless otherwise specified, this function should not modify the",(0,n.kt)("code",null,".grad")," field of the parameters."))))))))}p.isMDXComponent=!0}}]);