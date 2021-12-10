import React from "react";

const backmap = require('/docs/generated/api-mapping.json');

export default class ApiLink extends React.Component {

    url() {
        let identifier = this.props.name.replaceAll('.', '-')
        let pkg = this.package()
            .replace('classy.', '')
            .replaceAll('.', '/')

        return `/docs/api/${pkg}/#${identifier}`
    }

    package() {
        let pkg = this.props.package
        if (pkg === undefined) {
            let n = this.props.name.split('.')
            return backmap[n[0]]
        }
        return pkg
    }

    surface() {
        return this.props.displayName || this.props.name
    }

    render() {
        return (
            <a className={"direct-api-link"} title={this.props.name} href={this.url()} target={"_blank"}>
                {this.surface()}
            </a>
        );
    }
}
