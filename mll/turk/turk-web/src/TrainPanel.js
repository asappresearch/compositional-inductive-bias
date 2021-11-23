import React from 'react'
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form'
import Stack from 'react-bootstrap/Stack'


class TrainPanel extends React.Component {
    /*
    prop properties expected:
    - requesterId
    - taskId
    - serviceBaseUrl
    - numCards

    prop methods expected:
    - sendPost

    public methods:
    - nextExample()
    */
   state = {
       code: '',
       pictureUrl: ''
   }
    render = () => {
        return (
            <>
                <div>Here is an example of secret military device, and the code that goes with it</div>
                <img src={this.state.pictureUrl} alt="[You need to be able to view images]">
                </img>
                <br />
                <br />
                <Stack direction="vertical" gap={3}>
                    <Form.Group className="mb-3">
                        <Form.Label>Secret code:</Form.Label>
                        <Form.Control
                            type="text"
                            value={this.state.code}
                            style={{width: "150px"}}
                            readOnly={true}
                        />
                    </Form.Group>
                    <Button
                        variant="primary"
                        autoFocus
                        onClick={this.nextExample}
                    >
                        Next
                    </Button>
                </Stack>
                <br />
            </>
        );
    }
    nextExample = (e) => {
        this.fetchNextExample();
    }
    fetchNextExample = () => {
        console.log('train panel next example');
        this.props.sendPost(this.props.serviceBaseUrl + '/api/v1/fetch_training_example', {
            requesterId: this.props.requesterId,
            taskId: this.props.taskId,
            numCards: this.props.numCards
        })
        .then((response) => {
            console.log('got training example response', response);
            switch(response.messageType) {
                case 'example':
                    this.onReceivedExample(response);
                    break;
                case 'error':
                    console.log('error', response.error)
                    break;
                default:
                    return;
            }
        });
    }
    onReceivedExample = (response) => {
        this.setState({
            pictureUrl: this.props.serviceBaseUrl + '/' + response.pictureUrl,
            code: response.utt
        })
    }
}
export default TrainPanel
