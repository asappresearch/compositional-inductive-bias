import React from "react"
import './App.css';
import queryString from 'query-string';
import Tooltip from 'react-bootstrap/Tooltip'
import OverlayTrigger from 'react-bootstrap/OverlayTrigger'
import Tabs from 'react-bootstrap/Tabs'
import Tab from 'react-bootstrap/Tab'
import Card from 'react-bootstrap/Card'
import { v4 as uuidv4 } from 'uuid';

import EndGameModal from "./EndGameModal";
import TestPanel from "./TestPanel";
import Codes from "./Codes";
import TrainPanel from "./TrainPanel";


function sendPost(endPointUrl, request) {
    return new Promise(function(success, fail) {
        fetch(endPointUrl, {
            method: 'POST',
            headers: {'Content-Type': 'applications/json'},
            body: JSON.stringify(request)
        })
        .then(response => response.json())
        .then(response => {
            success(response);
        })
    });
}


class MouseOver extends React.Component {
    /*
    expected props:
    - text
    - innerJsx (works with spans)
    */
    render = () => {
        return (
            <OverlayTrigger overlay={
                <Tooltip>{this.props.text}</Tooltip>
            }
            >{this.props.innerJsx}</OverlayTrigger>
        );
    }
}


class App extends React.Component {
    state = {
        requesterId: uuidv4(),
        taskId: '',
        exampleIdx: 0,
        pictureUrl: '',
        score: 0,
        serviceBaseUrl: 'https://[REDACTED]',
        showEndGameModal: false,
        completionCode: '',
        finalScore: 0,
        totalSteps: 0,
        numCards: 2,
        maxCards: 10,
        cents_per_ten: 0
    }
    componentDidMount = () => {
        console.log('did mount');
        console.log('window.location.href', window.location.href);
        let serviceBaseUrl = '';
        if(window.location.href.indexOf('localhost') >= 0 ) {
            serviceBaseUrl = 'http://localhost:8000'
        } else {
            serviceBaseUrl = 'https://[REDACTED]'
        }
        let params = queryString.parse(window.location.search);
        console.log('params', params);
        this.setState({
            taskId: params.taskId,
            serviceBaseUrl: serviceBaseUrl
        })
        var requesterId = this.state.requesterId;
        if(params.requesterId != null) {
            console.log('have requesterId');
            requesterId = params.requesterId
            this.setState({
                requesterId: params.requesterId
            })
        }
        this.fetchTask({
            serviceBaseUrl: serviceBaseUrl,
            requesterId: requesterId,
            taskId: params.taskId
        });
    }
    render = () => {
        var new_style = {color: 'red', 'bold': true}
        return (
            <div style={{margin: '50px', width: '600px'}}>
                <audio id="rise04">
                    <source src="Rise04.mp3"></source>
                </audio>
                <audio id="coin01">
                    <source src="Coin01.mp3"></source>
                </audio>
                <audio id="upper01">
                    <source src="Upper01.mp3"></source>
                </audio>
                { this.state.showEndGameModal ?
                    <EndGameModal
                        completionCode={this.state.completionCode}
                        finalScore={this.state.finalScore}
                        sendFeedback={this.sendFeedback}
                    />
                     : null
                }

                <Card bg="info">
                    <Card.Body>
                        <Card.Title>Instructions</Card.Title>
                        <Card.Text>
                        You are a spy tasked with sending the blueprints for secret weaponry using secret codes.
                        First you need to study the codes, then you can use the code to transmit actual blueprints.
                        Don't worry if you get some wrong. Just try your best.
                        <br />
                        <br />
                        You can use the 'Train' tab to learn the different codes, then use them for real in the 'Test' tab.
                        You can use 'Add code' to get more codes.
                        <br />
                        <br />
                        </Card.Text>
                    </Card.Body>
                </Card>
                        <br />
                {this.state.cents_per_ten > 0 ? (
                        <span style={new_style}>New! {"You will get " + this.state.cents_per_ten + " cents bonus for every 10 points!"}</span>
                        ) : null
                        }
                        <br />
                <br />
                <Card bg="light">
                    <Card.Body>
                        <Card.Title>Status</Card.Title>
                        <Card.Text>
                            <MouseOver text={"You will get " + this.state.cents_per_ten + " cents for every 10 points."} innerJsx={
                                <span>Score: {this.state.score}</span>
                            }/><br />
                            <MouseOver text="Add more codes for more points." innerJsx={
                                <span>Points for each correct answer: {this.state.numCards - 1}</span>
                            }/><br />
                            <MouseOver text="If you are on MTurk, you need to complete all examples to get paid." innerJsx={
                                <span>Example: {this.state.exampleIdx + 1}/{this.state.totalSteps}</span>
                            }/><br />
                            {/* Requester ID: {this.state.requesterId} */}
                        </Card.Text>
                    </Card.Body>
                </Card>

                <br/>
                <Tabs defaultActiveKey="test" className="mb-3" onSelect={this.onTabSelected}>
                    <Tab eventKey="test" title="Test">
                        <TestPanel
                            ref={(c) => {this.testPanel = c}}
                            pictureUrl={this.state.pictureUrl}
                            exampleIdx={this.state.exampleIdx}
                            score={this.state.score}

                            sendCode={this.sendCode}
                            nextExample={this.fetchNextExample}
                        />
                    </Tab>
                    <Tab eventKey="train" title="Train">
                        <TrainPanel
                            ref={(c) => {this.trainPanel = c}}
                            requesterId={this.state.requesterId}
                            taskId={this.state.taskId}
                            sendPost={sendPost}
                            serviceBaseUrl={this.state.serviceBaseUrl}
                            numCards={this.state.numCards}
                        />
                    </Tab>
                </Tabs>
                    <Codes
                        numCards={this.state.numCards}
                        addCard={this.addCard}
                        removeCard={this.removeCard}
                    />
            </div>
        );
    }
    onTabSelected = (e) => {
        console.log('on tab selected', e);
        if(e === 'train') {
            console.log(' train tab selected');
            this.trainPanel.fetchNextExample();
        }
    }
    removeCard = (e) => {
        sendPost(this.state.serviceBaseUrl + '/api/v1/remove_card', {
            requesterId: this.state.requesterId,
            taskId: this.state.taskId
        })
        .then(response => {
            this.setState({
                numCards: response.numCards
            })
        })
    }
    addCard = (e) => {
        sendPost(this.state.serviceBaseUrl + '/api/v1/add_card', {
            requesterId: this.state.requesterId,
            taskId: this.state.taskId
        })
        .then(response => {
            this.setState({
                numCards: response.numCards
            })
        })
    }
    handleFetchResponse = (response) => {
        console.log('got server response', JSON.stringify(response));
        console.log('response.messagetype', response.messageType);
        switch(response.messageType) {
            case 'example':
                this.onReceivedExample(response);
                break
            case 'gameCompleted':
                this.onGameCompleted(response);
                break
            default:
        }
    }
    onReceivedExample = (response) => {
        let pictureUrl = this.state.serviceBaseUrl + '/' + response.pictureUrl;
        console.log('pictureUrl', pictureUrl);
        this.setState({
            pictureUrl: pictureUrl,
            exampleIdx: response.exampleIdx,
            score: response.score,
            totalSteps: response.totalSteps,
            maxCards: response.maxCards,
            numCards: response.numCards,
            cents_per_ten: response.cents_per_ten
        });
        this.testPanel.hideAlert();
        this.testPanel.focusCodeInput();
    }
    onGameCompleted = (response) => {
        console.log('got game completed message');
        document.getElementById('upper01').play();
        this.setState({
            completionCode: response.completionCode,
            finalScore: response.score,
        })
        this.showEndGameModal()
    }
    fetchTask = ({serviceBaseUrl, requesterId, taskId}) => {
        let request = {requesterId: requesterId, taskId: taskId, numCards: this.state.numCards}
        sendPost(serviceBaseUrl + '/api/v1/fetch_task', request)
        .then(response => {
            this.handleFetchResponse(response);
        })
    }
    fetchNextExample = (e) => {
        console.log("click next example");
        let request = {
            requesterId: this.state.requesterId, taskId: this.state.taskId, numCards: this.state.numCards}
        sendPost(this.state.serviceBaseUrl + '/api/v1/fetch_next', request)
        .then(response => {
            this.testPanel.disableNext();
            this.testPanel.clearCode();
            this.handleFetchResponse(response);
        })
    }
    showEndGameModal = () => {
        this.setState({
            showEndGameModal: true
        })
    }
    sendCode = (code) => {
        let request = {
            requesterId: this.state.requesterId,
            taskId: this.state.taskId,
            exampleIdx: this.state.exampleIdx,
            code: code
        }
        sendPost(this.state.serviceBaseUrl + '/api/v1/evaluate', request)
        .then((response) => {
            console.log("got evaluate response", response);
            let alertVariant = response.exampleCorrect === 1 ? 'success' : 'secondary';
            if(response.exampleCorrect === 1) {
                document.getElementById('rise04').play();
            } else {
                document.getElementById('coin01').play();
            }
            console.log('alertVariant', alertVariant);
            this.setState({
                score: response.score,
            })
            this.testPanel.showAlert({variant: alertVariant, message: response.resultText});
            this.testPanel.enableNext();
            this.testPanel.focusNext();
        });
    }
    sendFeedback = (feedbackText) => {
        console.log('app.js sendfeedback', feedbackText);
        sendPost(this.state.serviceBaseUrl + '/api/v1/send_feedback', {
            requesterId: this.state.requesterId,
            taskId: this.state.taskId,
            feedback: feedbackText
        })
    }
}

export default App;
