---
sidebar_position: 1
title: Tasks
---

import 'bootstrap/dist/css/bootstrap.css';

import { useState } from 'react'
import Container from 'react-bootstrap/Container';
import Card from 'react-bootstrap/Card';
import Button from 'react-bootstrap/Button';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Modal from 'react-bootstrap/Modal';

import ReactTermynal from '../../../src/components/termynal';

In its simplest flavor, classy lets **you focus entirely on your data** and automatically handles the development
of full-fledged models for you, with no ML knowledge or additional line of code required. To achieve this, classy only 
asks you to specify the task you wish to tackle and to organize your data into suitable formats.

export const ButtonWithBackdrop = ({children, title}) => {
    const [show, setShow] = useState(false)
    const handleClose = () => setShow(false)
    const handleShow = () => setShow(true)
    return (
        <div className="mt-auto">
            <div style={{textAlign: "center"}}>
                <Button className="mt-auto" onClick={handleShow}>Show Example</Button>
            </div>
            <Modal show={show} onHide={handleClose}>
                <Modal.Header>
                    <Modal.Title>{title}</Modal.Title>
                </Modal.Header>
                <Modal.Body>{children}</Modal.Body>
            </Modal>
        </div>
    )
}

There are five tasks currently supported by classy:
<Container>
    <Row>
        <Col sm={6}>
            <Card className="h-100">
                <Card.Header>Sequence Classification</Card.Header>
                <Card.Body className="d-flex flex-column">
                    <Card.Text style={{textAlign: "center"}}>Given a text in input (e.g. a sentence, a document), determine its most suitable label from a predefined set</Card.Text>
                    <ButtonWithBackdrop title="Example: Sentiment Analysis">
                        <Row style={{alignItems: "center"}}>
                            <Col sm={7} style={{textAlign: "center"}}>I love these headphones!</Col>
                            <Col sm={1} style={{textAlign: "center"}}> &#10132; </Col>
                            <Col sm={4} style={{textAlign: "center"}}> Positive </Col>
                        </Row>
                    </ButtonWithBackdrop>
                </Card.Body>
            </Card>
        </Col>
        <Col sm={6}>
            <Card className="h-100">
                <Card.Header>Sentence-Pair Classification</Card.Header>
                <Card.Body className="d-flex flex-column">
                    <Card.Text style={{textAlign: "center"}}>Given two texts in input (e.g. two sentences), determine the most suitable label for this pair (usually denoting some semantic relations) from a predefined set</Card.Text>
                    <ButtonWithBackdrop title="Example: Paraphrasis Detection">
                        <Row style={{alignItems: "center"}}>
                            <Col sm={7}>
                                <Row><div style={{textAlign: "center"}}><b>1.</b> I love these headphones!</div></Row>
                                <Row></Row>
                                <Row></Row>
                                <Row></Row>
                                <Row><div style={{textAlign: "center"}}><b>2.</b> I really like these headphones!</div></Row>
                            </Col>
                            <Col sm={1} style={{textAlign: "center"}}> &#10132; </Col>
                            <Col sm={4} style={{textAlign: "center"}}> Equivalent </Col>
                        </Row>
                    </ButtonWithBackdrop>
                </Card.Body>
            </Card>
        </Col>
    </Row>
    <Row style={{"marginTop": "1.5em"}}>
        <Col sm={6}>
            <Card className="h-100">
                <Card.Header>Token Classification</Card.Header>
                <Card.Body className="d-flex flex-column">
                    <Card.Text style={{textAlign: "center"}}>Given a list of tokens, for each of them, determine its most suitable label from a predefined set</Card.Text>
                    <ButtonWithBackdrop title="Example: Part-of-Speech Tagging">
                        <Row style={{alignItems: "center"}}>
                            <Col sm={3}>
                                <Row><div style={{textAlign: "center"}}>I</div></Row>
                                <Row></Row>
                                <Row><div style={{textAlign: "center"}}>&#129147;</div></Row>
                                <Row></Row>
                                <Row><div style={{textAlign: "center"}}>PRON</div></Row>
                            </Col>
                            <Col sm={3}>
                                <Row><div style={{textAlign: "center"}}>love</div></Row>
                                <Row></Row>
                                <Row><div style={{textAlign: "center"}}>&#129147;</div></Row>
                                <Row></Row>
                                <Row><div style={{textAlign: "center"}}>VERB</div></Row>
                            </Col>
                            <Col sm={3}>
                                <Row><div style={{textAlign: "center"}}>these</div></Row>
                                <Row></Row>
                                <Row><div style={{textAlign: "center"}}>&#129147;</div></Row>
                                <Row></Row>
                                <Row><div style={{textAlign: "center"}}>DET</div></Row>
                            </Col>
                            <Col sm={3}>
                                <Row><div style={{textAlign: "center"}}>headphones</div></Row>
                                <Row></Row>
                                <Row><div style={{textAlign: "center"}}>&#129147;</div></Row>
                                <Row></Row>
                                <Row><div style={{textAlign: "center"}}>NOUN</div></Row>
                            </Col>
                        </Row>
                    </ButtonWithBackdrop>
                </Card.Body>
            </Card>
        </Col>
        <Col sm={6}>
            <Card className="h-100">
                <Card.Header>Text Extraction</Card.Header>
                <Card.Body className="d-flex flex-column">
                    <Card.Text style={{textAlign: "center"}}>Given a context (e.g. a document) and some query about this document (e.g. a question), extract the consecutive span in the context that best addresses the query (e.g. an answer).</Card.Text>
                    <ButtonWithBackdrop title="Example: English-to-Italian Machine Translation">
                        <Row style={{alignItems: "center"}}>
                            <Col sm={5} style={{textAlign: "center"}}>
                                <p><b>Context:</b></p>
                                <p>I love these headphones!</p>
                                <p><b>Query:</b></p>
                                <p>What do you love?</p>
                            </Col>
                            <Col sm={2} style={{textAlign: "center"}}> &#10132; </Col>
                            <Col sm={5} style={{textAlign: "center"}}>I love <mark style={{backgroundColor: "#f1e740"}}>these headphones</mark>!</Col>
                        </Row>
                    </ButtonWithBackdrop>
                </Card.Body>
            </Card>
        </Col>
    </Row>
    <Row style={{"marginTop": "1.5em"}}>
        <Col sm={6}>
            <Card className="h-100">
                <Card.Header>Generation</Card.Header>
                <Card.Body className="d-flex flex-column">
                    <Card.Text style={{textAlign: "center"}}>Given a source text in input (e.g. a sentence), generate the corresponding target text.</Card.Text>
                    <ButtonWithBackdrop title="Example: English-to-Italian Machine Translation">
                        <Row style={{alignItems: "center"}}>
                            <Col sm={5} style={{textAlign: "center"}}>I love these headphones!</Col>
                            <Col sm={2} style={{textAlign: "center"}}> &#10132; </Col>
                            <Col sm={5} style={{textAlign: "center"}}>Io adoro queste cuffie!</Col>
                        </Row>
                    </ButtonWithBackdrop>
                </Card.Body>
            </Card>
        </Col>
    </Row>
</Container>

<p></p>

:::info

These tasks cover the vast majority of NLP problems.

:::
