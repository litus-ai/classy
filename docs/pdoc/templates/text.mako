<%
import os
import pdoc
from pdoc.html_helpers import glimpse, to_html as _to_html, format_git_link

def link(dobj: pdoc.Doc, name=None):
    name = name or dobj.qualname + ('()' if isinstance(dobj, pdoc.Function) else '')
    if isinstance(dobj, pdoc.External) and not external_links:
        return name
    qname = rel_refname(dobj, replace_dots=False)
    hname = qname
    mname = dobj.module.name.replace('classy.', 'api.').replace('.', '/')
    if dobj.module == dobj:
        hname = ''
        if len(dobj.submodules()) > 0:
            mname += '/index/'
    url = f"/docs/{mname}#{hname}"
    return '<a title="{}" href="{}">{}</a>{}'.format(qname, url, qname, '&nbsp;*' if mname.endswith('index/') else '')

def to_html(text):
    return _to_html(text, docformat=docformat, module=module, link=link, latex_math=latex_math).replace('<h2 ','<h5 ').replace('</h2>','</h5>').replace('<strong>','').replace('</strong>', '')

def get_annotation(bound_method, sep=':'):
    if isinstance(bound_method, pdoc.Variable):
        annot = show_type_annotations and bound_method.type_annotation(link=link) or ''
    else:
        annot = show_type_annotations and bound_method(link=link) or ''
    if annot:
        annot = ' ' + sep + '\N{NBSP}' + annot
    return annot

def rel_refname(o, is_class_init=False, replace_dots=True):
    res = o.qualname.replace('<locals>.', '') + ('.init' if is_class_init else '')
    if replace_dots:
        return res.replace('.', '-')
    else:
        return res

# adapted from https://stackoverflow.com/a/2020083/1908499
def fullname(klass):
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__
%>


<%def name="get_links(d, is_class_init=False)">
  % if d.obj is not getattr(d.inherits, 'obj', None):
  <div className="links-div">
      <a href="#${rel_refname(d, is_class_init=is_class_init)}" className="direct-link">#</a>
      <a href="${format_git_link(git_link_template, d)}" className="git-link">#</a>
  </div>
  %endif
</%def>

<%def name="ident(name)"><span className="ident">${name}</span></%def>

<%def name="show_desc(d, short=False)">
<%
inherits = ' inherited' if d.inherits else ''
docstring = glimpse(d.docstring) if short or inherits else d.docstring
%>
% if d.inherits:
*Inherited from:*
% if hasattr(d.inherits, 'cls'):
<code>${link(d.inherits.cls)}</code>.<code>${link(d.inherits, d.name)}</code>
% else:
<code>${link(d.inherits)}</code>
% endif
% endif
${docstring | to_html}
</%def>

<%def name="render_variable(var: pdoc.Variable)">
<div className="class-variable"><code className="name">
${ident('self' if var.instance_var else 'cls')}.${var.name}${get_annotation(var)}
</code></div>
</%def>

<%def name="render_mro(c: pdoc.Class)">
<%
all_mro = c.mro(only_documented=False)
direct_mro = c.mro(only_documented=True)
base_classes = set(fullname(e) for e in c.obj.__bases__)
mro = direct_mro if len(direct_mro) > 0 else all_mro
%>
% if len(mro) == 0:
class ${ident(c.name)}()
% else:
class ${ident(c.name)}(${', '.join(link(e) for e in mro if e.name in base_classes)})
% endif
</%def>

<%def name="show_module(module)">
<%
variables = module.variables(sort=sort_identifiers)
classes = module.classes(sort=sort_identifiers)
functions = module.functions(sort=sort_identifiers)
submodules = module.submodules()
%>
<%def name="show_func(f: pdoc.Function, is_method=False, annotation=None)">
<div className='api'>

% if is_method:
${"####"} ${f.name} ${"{#"}${rel_refname(f)}${"}"}
% else:
${"###"} ${f.name} ${"{#"}${rel_refname(f)}${"}"}
% endif
<div className='api__signature'>
% if annotation is not None:
<div className="annotation">@${annotation}</div>
% endif
% if len(f.params()) > 0:
${f.funcdef()} ${ident(f.name)}(<br/>
% for param in f.params(annotate=show_type_annotations, link=link):
&nbsp;&nbsp;&nbsp;&nbsp;${param},<br/>
% endfor
)${get_annotation(f.return_annotation, '\N{non-breaking hyphen}>')}
% else:
${f.funcdef()} ${ident(f.name)}()${get_annotation(f.return_annotation, '\N{non-breaking hyphen}>')}
% endif

${get_links(f)}
</div>

<div className='api__body'>
<div className='api__description'>
${show_desc(f)}
</div>
</div>
</div>
</%def>

${module.docstring | to_html}
% if submodules:
<h2 className="section-title" id="header-submodules">Sub-modules</h2>
<dl>
% for m in submodules:
% if not m.qualname == 'classy.version':
<dt>${link(m)}</dt>
<dd>${show_desc(m, short=True)}</dd>
% endif
% endfor
</dl>
% endif
% if variables:
<h2 className="section-title" id="header-variables">Global variables</h2>
<dl>
% for v in variables:
<% return_type = get_annotation(v.type_annotation) %>
<dt id="${v.refname}"><code className="name">var ${ident(v.name)}${return_type}</code></dt>
<dd>${show_desc(v)}</dd>
% endfor
</dl>
% endif
% if functions:
${"##"} Functions
% for f in functions:
${show_func(f)}
% endfor
% endif
% if classes:
${"## Classes {#clzs}"}
% for c in classes:
<%
## class_vars = c.class_variables(False, sort=sort_identifiers)
## inst_vars = c.instance_variables(False, sort=sort_identifiers)
## _vars = class_vars + inst_vars
smethods = c.functions(show_inherited_members, sort=sort_identifiers)
methods = c.methods(show_inherited_members, sort=sort_identifiers)
subclasses = c.subclasses()
%>

<div className='api'>

${"###"} ${c.name} ${"{#"}${rel_refname(c)}${"}"}

<div className='api__signature'>
${render_mro(c)}
${get_links(c)}
</div>

<div className='api__body'>
<div className='api__description'>

${show_desc(c)}

</div>

% if subclasses:
<details>
<summary>Subclasses (${len(subclasses)})</summary>
<div>

% for sub in subclasses:
- ${link(sub)}
% endfor

</div>
</details>

% endif

% if len(c.params()) > 1:
${"#### &#95;&#95;init&#95;&#95;"} ${"{#"}${rel_refname(c, is_class_init=True)}${"}"}
<div className='api__signature'>
def ${ident('__init__')}(<br/>
% for param in c.params(annotate=True, link=link):
&nbsp;&nbsp;&nbsp;&nbsp;${param},<br/>
% endfor
)
${get_links(c, is_class_init=True)}

</div>
% endif
<div className='api__description'></div>

## % if len(_vars) > 0:
## % for variable in class_vars + inst_vars:
## ${render_variable(variable)}
## % endfor
## % endif

% for f in methods:
${show_func(f, True)}
% endfor

% for f in smethods:
${show_func(f, True, annotation='staticmethod' if f.cls is None else 'classmethod')}
% endfor

% if show_inherited_members:
<%
members = c.inherited_members()
%>

% if members:
${"####"} Inherited members

% for cls, mems in members:
- ${link(cls)}:
% for m in mems:
- ${link(m, name=m.name)}
% endfor
% endfor
% endif
% endif

</div>
</div>
% endfor
% endif
</%def>

---
## title: ${'Namespace' if module.is_namespace else \
## 'Package' if module.is_package and not module.supermodule else \
## 'Module'} ${module.name}
title: ${module.name}
toc_min_heading_level: 2
toc_max_heading_level: 4
---

import '/src/css/api.css'
${show_module(module)}
