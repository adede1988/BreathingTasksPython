﻿#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on June 18, 2025, at 12:02
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'mindfulBreathing'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'trialTime': '120',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='G:\\My Drive\\GitHub\\BreathingTasksPython\\mindfulBreathing.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[2048, 1152], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instructions" ---
    instructionText = visual.TextStim(win=win, name='instructionText',
        text="In this task, you will be asked to count your breaths. Each trial will be 2 minutes long. Your goal is to count how many breaths you take in each 2 minute period. You will start each trial by pressing the spacebar, and the computer will let you know when to stop counting and enter the number of breaths that you counted. The goal here is to keep your awareness on your breath. The counting will help you do that by giving you a goal. There's no need to worry about controlling your breath, just observe and count.  \n\npress spacebar when you're ready to continue. ",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "getReadyToCount" ---
    getReadyInstructions = visual.TextStim(win=win, name='getReadyInstructions',
        text="Get ready to count your breaths. \n\npress the spacebar and begin counting when you're ready",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    startCountRsp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "countingPeriod" ---
    # Set experiment start values for variable component alreadySaved
    alreadySaved = False
    alreadySavedContainer = []
    # Set experiment start values for variable component flashCount
    flashCount = 1
    flashCountContainer = []
    # Set experiment start values for variable component flashTim
    flashTim = 2
    flashTimContainer = []
    blackPhotoDiodeBox = visual.Rect(
        win=win, name='blackPhotoDiodeBox',
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(-.7, .45), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1, -1, -1], fillColor=[-1, -1, -1],
        opacity=None, depth=-4.0, interpolate=True)
    whitePhotoDiode_1 = visual.Rect(
        win=win, name='whitePhotoDiode_1',
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(-.7, .45), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=None, fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    breathCountOnScreen = visual.TextStim(win=win, name='breathCountOnScreen',
        text='+\n\nCount your breaths :)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    
    # --- Initialize components for Routine "countInput" ---
    # Set experiment start values for variable component typedText
    typedText = ''
    typedTextContainer = []
    key_countRsp = keyboard.Keyboard()
    typedDisplay = visual.TextStim(win=win, name='typedDisplay',
        text='',
        font='Open Sans',
        pos=(0, -.2), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions.started', globalClock.getTime())
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # keep track of which components have finished
    instructionsComponents = [instructionText, key_resp]
    for thisComponent in instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructionText* updates
        
        # if instructionText is starting this frame...
        if instructionText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructionText.frameNStart = frameN  # exact frame index
            instructionText.tStart = t  # local t and not account for scr refresh
            instructionText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructionText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructionText.started')
            # update status
            instructionText.status = STARTED
            instructionText.setAutoDraw(True)
        
        # if instructionText is active this frame...
        if instructionText.status == STARTED:
            # update params
            pass
        
        # if instructionText is stopping this frame...
        if instructionText.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > instructionText.tStartRefresh + 1100000-frameTolerance:
                # keep track of stop time/frame for later
                instructionText.tStop = t  # not accounting for scr refresh
                instructionText.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instructionText.stopped')
                # update status
                instructionText.status = FINISHED
                instructionText.setAutoDraw(False)
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions.stopped', globalClock.getTime())
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=5.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "getReadyToCount" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('getReadyToCount.started', globalClock.getTime())
        startCountRsp.keys = []
        startCountRsp.rt = []
        _startCountRsp_allKeys = []
        # keep track of which components have finished
        getReadyToCountComponents = [getReadyInstructions, startCountRsp]
        for thisComponent in getReadyToCountComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "getReadyToCount" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *getReadyInstructions* updates
            
            # if getReadyInstructions is starting this frame...
            if getReadyInstructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                getReadyInstructions.frameNStart = frameN  # exact frame index
                getReadyInstructions.tStart = t  # local t and not account for scr refresh
                getReadyInstructions.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(getReadyInstructions, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'getReadyInstructions.started')
                # update status
                getReadyInstructions.status = STARTED
                getReadyInstructions.setAutoDraw(True)
            
            # if getReadyInstructions is active this frame...
            if getReadyInstructions.status == STARTED:
                # update params
                pass
            
            # if getReadyInstructions is stopping this frame...
            if getReadyInstructions.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > getReadyInstructions.tStartRefresh + 10000-frameTolerance:
                    # keep track of stop time/frame for later
                    getReadyInstructions.tStop = t  # not accounting for scr refresh
                    getReadyInstructions.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'getReadyInstructions.stopped')
                    # update status
                    getReadyInstructions.status = FINISHED
                    getReadyInstructions.setAutoDraw(False)
            
            # *startCountRsp* updates
            waitOnFlip = False
            
            # if startCountRsp is starting this frame...
            if startCountRsp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                startCountRsp.frameNStart = frameN  # exact frame index
                startCountRsp.tStart = t  # local t and not account for scr refresh
                startCountRsp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(startCountRsp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'startCountRsp.started')
                # update status
                startCountRsp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(startCountRsp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(startCountRsp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if startCountRsp.status == STARTED and not waitOnFlip:
                theseKeys = startCountRsp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _startCountRsp_allKeys.extend(theseKeys)
                if len(_startCountRsp_allKeys):
                    startCountRsp.keys = _startCountRsp_allKeys[-1].name  # just the last key pressed
                    startCountRsp.rt = _startCountRsp_allKeys[-1].rt
                    startCountRsp.duration = _startCountRsp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in getReadyToCountComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "getReadyToCount" ---
        for thisComponent in getReadyToCountComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('getReadyToCount.stopped', globalClock.getTime())
        # check responses
        if startCountRsp.keys in ['', [], None]:  # No response was made
            startCountRsp.keys = None
        trials.addData('startCountRsp.keys',startCountRsp.keys)
        if startCountRsp.keys != None:  # we had a response
            trials.addData('startCountRsp.rt', startCountRsp.rt)
            trials.addData('startCountRsp.duration', startCountRsp.duration)
        # the Routine "getReadyToCount" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "countingPeriod" ---
        continueRoutine = True
        # update component parameters for each repeat
        flashTim = 2  # Set Routine start values for flashTim
        # Run 'Begin Routine' code from diodeTimer
        flashCount = 1
        thisExp.addData('countingPeriod.started', globalClock.getTime())
        whitePhotoDiode_1.setFillColor([1,1,1])
        # keep track of which components have finished
        countingPeriodComponents = [blackPhotoDiodeBox, whitePhotoDiode_1, breathCountOnScreen]
        for thisComponent in countingPeriodComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "countingPeriod" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from diodeTimer
            if whitePhotoDiode_1.status == FINISHED: 
                whitePhotoDiode_1.status = NOT_STARTED
                flashTim = flashTim + 2 + random()
                alreadySaved = False
                
            if alreadySaved == False and whitePhotoDiode_1.status == NOT_STARTED and tThisFlip >= 2 - frameTolerance:
                thisExp.timestampOnFlip(win, 'diode' + str(flashCount))
                flashCount = flashCount + 1
                alreadySaved = True
                
              #how long should the trial be:   
            if t >= 10: 
                continueRoutine = False
            
            
            # *blackPhotoDiodeBox* updates
            
            # if blackPhotoDiodeBox is starting this frame...
            if blackPhotoDiodeBox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blackPhotoDiodeBox.frameNStart = frameN  # exact frame index
                blackPhotoDiodeBox.tStart = t  # local t and not account for scr refresh
                blackPhotoDiodeBox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blackPhotoDiodeBox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blackPhotoDiodeBox.started')
                # update status
                blackPhotoDiodeBox.status = STARTED
                blackPhotoDiodeBox.setAutoDraw(True)
            
            # if blackPhotoDiodeBox is active this frame...
            if blackPhotoDiodeBox.status == STARTED:
                # update params
                pass
            
            # if blackPhotoDiodeBox is stopping this frame...
            if blackPhotoDiodeBox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blackPhotoDiodeBox.tStartRefresh + 120-frameTolerance:
                    # keep track of stop time/frame for later
                    blackPhotoDiodeBox.tStop = t  # not accounting for scr refresh
                    blackPhotoDiodeBox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blackPhotoDiodeBox.stopped')
                    # update status
                    blackPhotoDiodeBox.status = FINISHED
                    blackPhotoDiodeBox.setAutoDraw(False)
            
            # *whitePhotoDiode_1* updates
            
            # if whitePhotoDiode_1 is starting this frame...
            if whitePhotoDiode_1.status == NOT_STARTED and tThisFlip >= flashTim-frameTolerance:
                # keep track of start time/frame for later
                whitePhotoDiode_1.frameNStart = frameN  # exact frame index
                whitePhotoDiode_1.tStart = t  # local t and not account for scr refresh
                whitePhotoDiode_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(whitePhotoDiode_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'whitePhotoDiode_1.started')
                # update status
                whitePhotoDiode_1.status = STARTED
                whitePhotoDiode_1.setAutoDraw(True)
            
            # if whitePhotoDiode_1 is active this frame...
            if whitePhotoDiode_1.status == STARTED:
                # update params
                pass
            
            # if whitePhotoDiode_1 is stopping this frame...
            if whitePhotoDiode_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > whitePhotoDiode_1.tStartRefresh + .20-frameTolerance:
                    # keep track of stop time/frame for later
                    whitePhotoDiode_1.tStop = t  # not accounting for scr refresh
                    whitePhotoDiode_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'whitePhotoDiode_1.stopped')
                    # update status
                    whitePhotoDiode_1.status = FINISHED
                    whitePhotoDiode_1.setAutoDraw(False)
            
            # *breathCountOnScreen* updates
            
            # if breathCountOnScreen is starting this frame...
            if breathCountOnScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                breathCountOnScreen.frameNStart = frameN  # exact frame index
                breathCountOnScreen.tStart = t  # local t and not account for scr refresh
                breathCountOnScreen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(breathCountOnScreen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'breathCountOnScreen.started')
                # update status
                breathCountOnScreen.status = STARTED
                breathCountOnScreen.setAutoDraw(True)
            
            # if breathCountOnScreen is active this frame...
            if breathCountOnScreen.status == STARTED:
                # update params
                pass
            
            # if breathCountOnScreen is stopping this frame...
            if breathCountOnScreen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > breathCountOnScreen.tStartRefresh + 120-frameTolerance:
                    # keep track of stop time/frame for later
                    breathCountOnScreen.tStop = t  # not accounting for scr refresh
                    breathCountOnScreen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'breathCountOnScreen.stopped')
                    # update status
                    breathCountOnScreen.status = FINISHED
                    breathCountOnScreen.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in countingPeriodComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "countingPeriod" ---
        for thisComponent in countingPeriodComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        
        
        thisExp.addData('flashTim.routineEndVal', flashTim)  # Save end Routine value
        thisExp.addData('countingPeriod.stopped', globalClock.getTime())
        # the Routine "countingPeriod" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "countInput" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code
        typedText = ''
        
        thisExp.addData('countInput.started', globalClock.getTime())
        key_countRsp.keys = []
        key_countRsp.rt = []
        _key_countRsp_allKeys = []
        # keep track of which components have finished
        countInputComponents = [key_countRsp, typedDisplay]
        for thisComponent in countInputComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "countInput" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code
            
            if key_countRsp.status == STARTED and not waitOnFlip:
                theseKeys = key_countRsp.getKeys(keyList=['return', 'num_enter', 'backspace', '0','1','2','3','4','5','6','7','8','9'], ignoreKeys=["escape"], waitRelease=False)
                             
            
                for key in theseKeys:
                    if key in ['return', 'num_enter']:
                        continueRoutine = False
                    elif key in ['backspace']:
                        typedText = typedText[:-1]
                    elif key in ['0','1','2','3','4','5','6','7','8','9']:
                        if len(typedText) < 2:  # limit to 2 digits
                            typedText += str(key.name)  
            
            # *key_countRsp* updates
            waitOnFlip = False
            
            # if key_countRsp is starting this frame...
            if key_countRsp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_countRsp.frameNStart = frameN  # exact frame index
                key_countRsp.tStart = t  # local t and not account for scr refresh
                key_countRsp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_countRsp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_countRsp.started')
                # update status
                key_countRsp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_countRsp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_countRsp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_countRsp.status == STARTED and not waitOnFlip:
                theseKeys = key_countRsp.getKeys(keyList=['return', 'num_enter', 'backspace', '0','1','2','3','4','5','6','7','8','9'], ignoreKeys=["escape"], waitRelease=False)
                _key_countRsp_allKeys.extend(theseKeys)
                if len(_key_countRsp_allKeys):
                    key_countRsp.keys = _key_countRsp_allKeys[-1].name  # just the last key pressed
                    key_countRsp.rt = _key_countRsp_allKeys[-1].rt
                    key_countRsp.duration = _key_countRsp_allKeys[-1].duration
            
            # *typedDisplay* updates
            
            # if typedDisplay is starting this frame...
            if typedDisplay.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                typedDisplay.frameNStart = frameN  # exact frame index
                typedDisplay.tStart = t  # local t and not account for scr refresh
                typedDisplay.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(typedDisplay, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'typedDisplay.started')
                # update status
                typedDisplay.status = STARTED
                typedDisplay.setAutoDraw(True)
            
            # if typedDisplay is active this frame...
            if typedDisplay.status == STARTED:
                # update params
                typedDisplay.setText(typedText, log=False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in countInputComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "countInput" ---
        for thisComponent in countInputComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        
        # Run 'End Routine' code from code
        thisExp.addData('typedResponse', typedText)
        thisExp.addData('countInput.stopped', globalClock.getTime())
        # check responses
        if key_countRsp.keys in ['', [], None]:  # No response was made
            key_countRsp.keys = None
        trials.addData('key_countRsp.keys',key_countRsp.keys)
        if key_countRsp.keys != None:  # we had a response
            trials.addData('key_countRsp.rt', key_countRsp.rt)
            trials.addData('key_countRsp.duration', key_countRsp.duration)
        # the Routine "countInput" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 5.0 repeats of 'trials'
    
    
    
    
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
