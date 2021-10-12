import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Fast Development',
    Svg: require('../../static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Whether for prototyping or production, reduce the time to get things up and running by leaps and bounds.
        From data to deployed systems in a matter of few commands.
      </>
    ),
  },
  {
    title: 'CLI Is All You Need',
    Svg: require('../../static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Classy is built around the command line interface, you can train, present and REST-expose powerful ML models
        without writing a single line of code just by executing bash commands.
      </>
    ),
  },
  {
    title: 'Data Centric',
    Svg: require('../../static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        In Classy data are central. Organize your datasets in common formats like TSV and JSONL and Classy will do the rest.
      </>
    ),
  },
];

function Feature({ Svg, title, description }) {
  return (
    <div className={clsx('col col--4')}>
      {/* <div className="text--center">
        <Svg className={styles.featureSvg} alt={title} />
      </div> */}
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
