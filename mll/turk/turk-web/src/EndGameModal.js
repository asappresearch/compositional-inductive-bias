import React from 'react'
import Modal from 'react-bootstrap/Modal'
import Form from 'react-bootstrap/Form'
import Card from 'react-bootstrap/Card'
import Button from 'react-bootstrap/Button';
import Alert from 'react-bootstrap/Alert'


class EndGameModal extends React.Component {
    /*
    props expected:
    - showEnGameModal
    - completionCode
    - finalScore

    prop methods expected
    - sendFeedback(feedbackText)
    */
    state = {
        feedbackText: '',
        alertShow: false
    }
    render = () => {
        return (
                <Modal
                    size="lg"
                    aria-labelledby="contained-modal-title-vcenter"
                    centered
                    show="true"
                    // show={this.props.showEndGameModal.toString()}
                >
                <Modal.Header>
                    <Modal.Title>
                    You have finished the task!
                    </Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    {/* <h4>You have finished the task!</h4> */}
                    <p>
                    Your completion code is: {this.props.completionCode}. Please paste this into the MTurk dialog, in order to submit the MTurk task.
                    </p>
                    <p>
                        Thank you for playing! Your final score was: {this.props.finalScore}. (Reminder: score is just for fun, not related to pay).
                    </p>
                    <br />
                    <br />
                    <br />
                <Card bg="light">
                    <Card.Body>
                        <Card.Title>Feedback</Card.Title>
                        <Card.Text>
                        If you have any comments, requests, or other feedback, please paste into the below box, and press Submit. We will review these comments off-line.
                    <br />
                    <br />
                        {this.state.alertShow ?
                        (<Alert variant='success' show={true}>
                            Thank you for your feedback!
                        </Alert>): null}
                        <Form.Control
                            as="textarea"
                            value={this.state.feedbackText}
                            onChange={this.feedbackTextChange}
                            placeholder="Leave a comment here"
                            style={{height: '100px'}}
                        />
                    <br />
                        <Button onClick={this.onSubmitFeedback}>Submit</Button>
                        </Card.Text>
                    </Card.Body>
                </Card>
                </Modal.Body>
                </Modal>
        );
    }
    feedbackTextChange = (e) => {
        this.setState({
            feedbackText: e.target.value
        })
    }
    onSubmitFeedback = (e) => {
        console.log('onsubmitfeedback');
        if(this.state.feedbackText === ''){
            return
        }
        this.props.sendFeedback(this.state.feedbackText)
        this.setState({
            feedbackText: '',
            alertShow: true
        })
        setTimeout(
            () => this.setState({
                alertShow: false
            }), 1000
        )
    }
}
export default EndGameModal
