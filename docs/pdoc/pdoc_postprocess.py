import json
from pathlib import Path
from typing import Set, Optional

import pdoc


def build_recursive(module, mapping):
    for element in module.classes() + module.functions():
        mapping.setdefault(element.name, []).append(module.name)

    for m in module.submodules():
        build_recursive(m, mapping)


def build_mapping(main_module: pdoc.Module):
    mapping = {}

    build_recursive(main_module, mapping)
    to_remove = set()

    for name, packages in mapping.items():
        if len(packages) > 1:
            to_remove.add(name)
            print("*" * 4, end=" ")
        print(name, "->", ", ".join(packages))

    for e in to_remove:
        del mapping[e]

    tgt_file = Path("docs/generated/api-mapping.json")
    tgt_file.parent.mkdir(parents=True, exist_ok=True)

    with tgt_file.open("w") as f:
        json.dump({name: packages[0] for name, packages in mapping.items()}, f, indent=2)


def module_name(module: pdoc.Module):
    if module.is_package:
        return module.name.split(".")[-1]

    return ".".join(module.name.split(".")[-2:])


def items_from_module(module: pdoc.Module, skip_modules: Optional[Set[str]] = None):
    skip_modules = skip_modules or set()

    items = []

    for m in module.submodules():
        if m.name in skip_modules:
            continue

        if m.is_package:
            items.append(items_from_module(m))
        else:
            """
            {
              type: 'doc',
              id: 'i18n/tutorial',
              label: 'Tutorial',
            }
            """
            items.append(
                dict(
                    type="doc",
                    id=m.name.replace("classy.", "api.").replace(".", "/"),
                    label=module_name(m),
                )
            )

    return dict(
        type="category",
        label=module_name(module),
        items=items,
    )


def build_sidebar(module):
    with open("docs/generated/api-sidebar.json", "w") as f:
        items = items_from_module(module, skip_modules={"classy.version"})["items"]
        items.insert(
            0,
            dict(
                type="doc",
                id="api/main",
                label="API Reference",
            ),
        )
        json.dump(items, f, indent=2)


def build_api_landing():
    content = """---
pagination_next: null
pagination_prev: null
title: classy Reference API
---

Welcome to `classy`'s reference API!

On the left side, you can find quick links for the package structure.
"""

    with open("docs/docs/api/main.md", "w") as f:
        f.write(content)


def clean_pdoc_files():
    generated_dir = Path("docs/docs/api")
    for file in generated_dir.rglob("**/index.md"):
        print("removing", file)
        file.unlink()


def main():
    ctx = pdoc.Context()
    main_module = pdoc.Module("classy", context=ctx)
    build_mapping(main_module)
    build_sidebar(main_module)
    build_api_landing()
    clean_pdoc_files()


if __name__ == "__main__":
    main()
