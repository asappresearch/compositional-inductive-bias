import React from 'react'
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Tooltip from 'react-bootstrap/Tooltip';
import Button from 'react-bootstrap/Button';
import Card from 'react-bootstrap/Card'


class Codes extends React.Component {
    /*
    prop properties expected:
    - numCards

    prop methods expected:
    - addCard
    - removeCard
    */
    render = () => {
        return (
            <>
                <Card bg="light">
                    <Card.Body>
                        <Card.Title>Codes</Card.Title>
                        <Card.Text>
                            You have {this.props.numCards} codes that you need to know. You can click on 'Add code' to get more. More codes means more points, but
                            makes task harder.
                            <br />
                            <br />
                            <OverlayTrigger overlay={
                            <Tooltip id="points-tooltip">Adding more codes increases your points for correct answers, but increases the difficulty.</Tooltip>
                            }
                            >
                            <span>Codes available: {this.props.numCards}</span>
                            </OverlayTrigger>
                            <br />
                            <br />
                            <Button variant="secondary" onClick={this.props.addCard}>Add code</Button> &nbsp;&nbsp;&nbsp;&nbsp;
                            <Button variant="secondary" onClick={this.props.removeCard}>Remove code</Button>
                        </Card.Text>
                    </Card.Body>
                </Card>
            </>
        );
    }
}
export default Codes
