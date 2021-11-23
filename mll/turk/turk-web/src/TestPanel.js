import React from 'react'
import Alert from 'react-bootstrap/Alert'
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form'


class TestPanel extends React.Component {
    /*
    prop properties expected:
    - exampleIdx
    - score

    prop methods expected:
    - sendCode
    - nextExample

    public methods:
    - focusCodeInput()
    - focusNext()
    - showAlert(variant, message)
    - hideAlert()
    - enableNext()
    - disableNext()
    - clearCode()
    */
   state = {
       code: '',
       alertVariant: '',
       alertShow: false,
       alertText: '',
       nextEnabled: false
   }
    render = () => {
        return (
            <>
                <div>Here is a secret military device you need to send by code:</div>
                <img src={this.props.pictureUrl} alt="[You need to be able to view images]">
                </img>
                <br />
                <br />
                <form onSubmit={e => e.preventDefault()}>
                Please type in your best guess of the secret code, then click Send<br />
                <br />
                <Alert variant={this.state.alertVariant} show={this.state.alertShow}>
                    {this.state.alertText}
                </Alert>
                <Form.Group className="mb-3">
                <Form.Label>Secret code:</Form.Label>
                <Form.Control
                    type="text"
                    value={this.state.code}
                    onChange={this.codeChange}
                    style={{width: "150px"}}
                    autoFocus
                    ref={(input) => {this.codeInput = input; }}
                ></Form.Control>
                <br />
                <Button
                    type="submit"
                    variant="primary"
                    onClick={this.clickSend}
                >Send</Button>
                &nbsp;&nbsp;&nbsp;&nbsp;
                <Button
                    variant="secondary"
                    ref={(input) => {this.nextButton = input;}}
                    disabled={!this.state.nextEnabled}
                    onClick={this.props.nextExample}
                >Next</Button>
                </Form.Group>
                <br />
                </form>
            </>
        );
    }
    focusCodeInput = () => {
        this.codeInput.focus();
    }
    focusNext = () => {
        this.nextButton.focus();
    }
    enableNext = () => {
        this.setState({
            nextEnabled: true
        })
    }
    disableNext = () => {
        this.setState({
            nextEnabled: false
        })
    }
    codeChange = (e) => {
        this.setState({
            code: e.target.value.toLowerCase()
        });
    }
    clearCode = () => {
        this.setState({
            code: ''
        })
    }
    clickSend = (e) => {
        console.log('clickSend', this.state.code);
        if(this.state.code === '') {
            return;
        }
        this.props.sendCode(this.state.code);
    }
    showAlert = ({variant, message}) => {
        this.setState({
            alertShow: true,
            alertVariant: variant,
            alertText: message
        })
    }
    hideAlert = () => {
        this.setState({
            alertShow: false
        })
    }
}
export default TestPanel
