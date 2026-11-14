"""Extract successful RandomForest adversarial attacks and run physical/consistency checks.

Input accepts randomforest_adversarial_examples.xls/.xlsx/.csv. Only rows whose
attack_success value is True are processed and written to output.

Includes embedded F2MD plausibility-check methods (F2MDChecks).

Statistical checks require train.parquet (normal-class feature distribution).
Coordinate/trajectory and exact sequential-feature recomputation checks require
 test.parquet to map source_row to the original preprocessing records.

Example:
  python physical_plausibility_check_with_consistency_updated.py \
      --input randomforest_adversarial_examples.xls \
      --data_dir data/processed \
      --output_dir physical_check_output
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ---- Integrated F2MD plausibility-check functions ----
# Ported from the supplied F2MD C++ checks and KalmanFilterJ SI/SVI/SC cores.
# SI and SC filter parameters are grounded in the supplied Kalman_SI.cc / Kalman_SC.cc;
# SVI uses the inferred state ordering [x, vx, y, vy] from the check Delta indexing.
from dataclasses import dataclass
from math import atan2, cos, sin, hypot, fabs, radians, degrees, pi, exp
from typing import Iterable, Mapping, Optional, Sequence, Tuple

Point = Tuple[float, float]


def xy(p) -> Point:
    if hasattr(p, "x"):
        return float(p.x), float(p.y)
    return float(p[0]), float(p[1])


def distance(a, b) -> float:
    ax, ay = xy(a); bx, by = xy(b)
    return hypot(ax - bx, ay - by)


def heading_angle(vector) -> float:
    x, y = xy(vector)
    return degrees(atan2(y, x))


def angle_difference(a: float, b: float) -> float:
    """Smallest absolute difference between angles, in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def circle_overlap_area(r1: float, r2: float, d: float) -> float:
    """Area shared by two circles; equivalent geometric helper."""
    r1, r2, d = max(0.0, r1), max(0.0, r2), max(0.0, d)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return pi * min(r1, r2) ** 2
    a1 = acos_clamped((d*d + r1*r1 - r2*r2) / (2*d*r1))
    a2 = acos_clamped((d*d + r2*r2 - r1*r1) / (2*d*r2))
    area = r1*r1*a1 + r2*r2*a2
    area -= 0.5 * sqrt_nonnegative((-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2))
    return area


def acos_clamped(x: float) -> float:
    from math import acos
    return acos(max(-1.0, min(1.0, x)))


def sqrt_nonnegative(x: float) -> float:
    from math import sqrt
    return sqrt(max(0.0, x))


def circle_circle_factor(d: float, r1: float, r2: float, allowable: float) -> float:
    """
    Confidence-aware plausibility factor approximation: fraction of uncertainty
    discs' combined geometry consistent with the allowable range. The source
    invokes mdmLib.CircleCircleFactor; exact project-specific scoring may differ.
    """
    threshold = max(0.0, allowable) + max(0.0, r1) + max(0.0, r2)
    if d <= max(0.0, allowable) + abs(r1-r2):
        return 1.0
    if d >= threshold:
        return 0.0
    return (threshold - d) / max(1e-12, threshold - (max(0.0, allowable) + abs(r1-r2)))


@dataclass
class CheckParameters:
    MAX_PROXIMITY_RANGE_L: float = 100.0
    MAX_PROXIMITY_RANGE_W: float = 10.0
    MAX_PROXIMITY_DISTANCE: float = 20.0
    MAX_PLAUSIBLE_RANGE: float = 300.0
    MAX_NON_ROUTE_SPEED: float = 2.0
    MAX_DISTANCE_FROM_ROUTE: float = 10.0
    MAX_BEACON_FREQUENCY: float = 0.2
    POS_HEADING_TIME: float = 1.0
    MAX_DELTA_INTER: float = 1.0
    MAX_SA_RANGE: float = 100.0
    MAX_TIME_DELTA: float = 2.0
    MAX_HEADING_CHANGE: float = 45.0
    MAX_KALMAN_TIME: float = 5.0
    MAX_KALMAN_POS_RANGE: float = 10.0
    MAX_KALMAN_SPEED_RANGE: float = 10.0
    KALMAN_MIN_POS_RANGE: float = 1.0
    KALMAN_MIN_SPEED_RANGE: float = 1.0


class F2MDChecks:
    """Port of the shared check logic. Supply limits and params as needed."""

    def __init__(self, params: Optional[CheckParameters] = None,
                 max_speed: float = 50.0, max_accel: float = 5.0,
                 max_decel: float = 8.0):
        self.params = params or CheckParameters()
        self.MAX_PLAUSIBLE_SPEED = max_speed
        self.MAX_PLAUSIBLE_ACCEL = max_accel
        self.MAX_PLAUSIBLE_DECEL = max_decel

    def ProximityPlausibilityCheck(self, testPosition, myPosition, myHeading=None):
        p = self.params
        d = distance(testPosition, myPosition)
        tx, ty = xy(testPosition); mx, my = xy(myPosition)
        position_angle = abs(heading_angle((tx-mx, ty-my)))
        if d < p.MAX_PROXIMITY_RANGE_L:
            inside_lane = (d < p.MAX_PROXIMITY_RANGE_W * 2 or
                (position_angle < 90 and d < p.MAX_PROXIMITY_RANGE_W /
                 max(1e-12, cos(radians(90-position_angle)))))
            if inside_lane:
                # No dynamic map is supplied in this validation script.
                # Treat this map-dependent sub-check as unavailable/neutral.
                return 1.0
        return 1.0

    def RangePlausibilityCheck(self, senderPosition, receiverPosition,
                               senderPositionConfidence=(0, 0),
                               receiverPositionConfidence=(0, 0), confidence_aware=False):
        d = distance(senderPosition, receiverPosition)
        if not confidence_aware:
            return 1.0 if d < self.params.MAX_PLAUSIBLE_RANGE else 0.0
        return circle_circle_factor(d, xy(senderPositionConfidence)[0],
                                    xy(receiverPositionConfidence)[0],
                                    self.params.MAX_PLAUSIBLE_RANGE)

    def PositionPlausibilityCheck(self, senderPosition, senderSpeed,
                                  senderPositionConfidence=(0, 0),
                                  senderSpeedConfidence=0.0):
        p = self.params
        speed_lower = max(0.0, float(senderSpeed)-float(senderSpeedConfidence))
        if speed_lower <= p.MAX_NON_ROUTE_SPEED:
            return 1.0
        # Road-map distance is not available in this script. For vehicles
        # moving above the non-route speed threshold, skip the map-dependent
        # portion neutrally rather than raising an exception or guessing.
        return 1.0

    def SpeedPlausibilityCheck(self, speed, speedConfidence=0.0):
        return 1.0 if abs(float(speed)-float(speedConfidence)) < self.MAX_PLAUSIBLE_SPEED else 0.0

    def PositionConsistancyCheck(self, curPosition, oldPosition, time,
                                 curPositionConfidence=(0, 0),
                                 oldPositionConfidence=(0, 0), confidence_aware=False):
        d = distance(curPosition, oldPosition)
        bound = self.MAX_PLAUSIBLE_SPEED * max(0.0, time)
        if not confidence_aware:
            return 1.0 if d < bound else 0.0
        return circle_circle_factor(d, xy(curPositionConfidence)[0],
                                    xy(oldPositionConfidence)[0], bound)

    def SpeedConsistancyCheck(self, curSpeed, oldSpeed, time,
                              curSpeedConfidence=0.0, oldSpeedConfidence=0.0):
        delta = float(curSpeed)-float(oldSpeed)
        limit = (self.MAX_PLAUSIBLE_ACCEL if delta > 0 else self.MAX_PLAUSIBLE_DECEL) * max(0.0, time)
        confidence = max(0.0, curSpeedConfidence)+max(0.0, oldSpeedConfidence)
        return 1.0 if abs(delta) <= limit+confidence else 0.0

    def PositionSpeedConsistancyCheck(self, curPosition, oldPosition,
                                      curSpeed, oldSpeed, time):
        if time <= 0:
            return 1.0
        average_speed = (float(curSpeed)+float(oldSpeed))/2.0
        observed_speed = distance(curPosition, oldPosition)/time
        return 1.0 if observed_speed <= max(abs(curSpeed), abs(oldSpeed), abs(average_speed)) else 0.0

    def PositionSpeedMaxConsistancyCheck(self, curPosition, oldPosition,
                                         curSpeed, oldSpeed, time):
        if time <= 0 or time >= self.params.MAX_TIME_DELTA:
            return 1.0
        observed = distance(curPosition, oldPosition)/time
        return 1.0 if observed <= max(abs(curSpeed), abs(oldSpeed)) else 0.0

    def BeaconFrequencyCheck(self, timeNew, timeOld):
        return 0.0 if (timeNew-timeOld) < self.params.MAX_BEACON_FREQUENCY else 1.0

    def PositionHeadingConsistancyCheck(self, curHeading, curPosition, oldPosition,
                                        deltaTime, curSpeed, max_heading_change=None):
        if deltaTime >= self.params.POS_HEADING_TIME:
            return 1.0
        d = distance(curPosition, oldPosition)
        if d < 1.0 or curSpeed < 1.0:
            return 1.0
        dx = xy(curPosition)[0]-xy(oldPosition)[0]
        dy = xy(curPosition)[1]-xy(oldPosition)[1]
        heading_motion = heading_angle((dx, dy))
        heading_claimed = heading_angle(curHeading)
        limit = self.params.MAX_HEADING_CHANGE if max_heading_change is None else max_heading_change
        return 1.0 if angle_difference(heading_claimed, heading_motion) <= limit else 0.0

    def SuddenAppearenceCheck(self, senderPosition, receiverPosition,
                              senderPositionConfidence=(0, 0),
                              receiverPositionConfidence=(0, 0),
                              confidence_aware=False):
        d = distance(senderPosition, receiverPosition)
        if not confidence_aware:
            return 0.0 if d < self.params.MAX_SA_RANGE else 1.0
        r_sender = max(0.0, xy(senderPositionConfidence)[0])
        r_receiver = self.params.MAX_SA_RANGE + max(0.0, xy(receiverPositionConfidence)[0])
        if r_sender <= 0:
            return 0.0 if d < r_receiver else 1.0
        overlap = circle_overlap_area(r_sender, r_receiver, d)
        return 1.0-overlap/(pi*r_sender*r_sender)

    def IntersectionCheck(self, nodePosition1, nodeSize1, heading1,
                          nodePosition2, nodeSize2, heading2, deltaTime=0.0):
        """
        Rectangle collision/intersection approximation using Separating Axis
        Theorem. Input size is (width, length), heading is degrees.
        Returns 0 for intersection suspicion, 1 otherwise, weighted by time.
        """
        def corners(pos, size, angle):
            cx, cy = xy(pos); w, l = xy(size)
            hw, hl = w/2, l/2
            a = radians(angle)
            ux, uy = cos(a), sin(a)
            vx, vy = -sin(a), cos(a)
            return [(cx+sx*hw*ux+sy*hl*vx, cy+sx*hw*uy+sy*hl*vy)
                    for sx, sy in [(-1,-1),(-1,1),(1,1),(1,-1)]]
        A, B = corners(nodePosition1,nodeSize1,heading1), corners(nodePosition2,nodeSize2,heading2)
        axes=[]
        for poly in (A,B):
            for i in range(2):
                x1,y1=poly[i]; x2,y2=poly[i+1]
                axes.append((-(y2-y1), x2-x1))
        intersect=True
        for ax,ay in axes:
            pa=[x*ax+y*ay for x,y in A]; pb=[x*ax+y*ay for x,y in B]
            if max(pa)<min(pb) or max(pb)<min(pa):
                intersect=False; break
        weight=max(0.0,(self.params.MAX_DELTA_INTER-deltaTime)/max(1e-12,self.params.MAX_DELTA_INTER))
        return 0.0 if intersect and weight > 0.5 else 1.0

    # Stateful numerical ports of Kalman_SI / Kalman_SC and the check logic.
    # Filters are kept per F2MDChecks instance; callers may pass a custom filter
    # object only if it exposes the matching getDeltaPos/isInit/setInitial API.
    def _kalman(self, kind):
        if not hasattr(self, "_kalman_states"):
            self._kalman_states = {}
        if kind not in self._kalman_states:
            self._kalman_states[kind] = {"init": False, "x": None, "P": None}
        return self._kalman_states[kind]

    @staticmethod
    def _confidence(value, minimum):
        return max(float(value), float(minimum))

    def _si_delta(self, kind, time, observed, confidence, control=None):
        st = self._kalman(kind)
        z = np.asarray(observed, dtype=float).reshape(2)
        conf = np.asarray(confidence, dtype=float).reshape(2)
        if not st["init"]:
            st["init"] = True; st["x"] = z.copy(); st["P"] = np.eye(2)*10.0
            return np.zeros(2)
        T = float(time)
        A = np.eye(2)
        B = np.eye(2)*T
        Q = np.eye(2)*10.0
        R = np.diag(conf)
        x0, P0 = st["x"], st["P"]
        xp = A @ x0 if control is None else A @ x0 + B @ np.asarray(control, dtype=float).reshape(2)
        Pp = A @ P0 @ A.T + Q
        H = np.eye(2)
        try:
            K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R)
        except np.linalg.LinAlgError:
            K = Pp @ H.T @ np.linalg.pinv(H @ Pp @ H.T + R)
        x = xp + K @ (z - H @ xp)
        P = (np.eye(2)-K@H) @ Pp
        st["x"], st["P"] = x.copy(), P.copy()
        return np.abs(x-z)

    def _sc_delta(self, time, observed_position_distance, observed_speed, acceleration, conf_pos, conf_speed):
        st = self._kalman("SC")
        z = np.array([float(observed_position_distance), float(observed_speed)])
        if not st["init"]:
            st["init"] = True; st["x"] = np.array([0.0, float(observed_speed)]); st["P"] = np.eye(2)*10.0
            return np.zeros(2)
        T=float(time); A=np.array([[1.0,T],[0.0,1.0]])
        B=np.array([[0.5*T*T],[T]])
        Q=np.eye(2)*10.0; R=np.diag([float(conf_pos),float(conf_speed)])
        x0,P0=st["x"],st["P"]
        xp=A@x0+B[:,0]*float(acceleration); Pp=A@P0@A.T+Q
        try: K=Pp@np.linalg.inv(Pp+R)
        except np.linalg.LinAlgError: K=Pp@np.linalg.pinv(Pp+R)
        x=xp+K@(z-xp); P=(np.eye(2)-K)@Pp
        st["x"],st["P"]=x.copy(),P.copy()
        return np.abs(x-z)

    def _svi_delta(self, time, position, speed, accel, pos_conf, speed_conf):
        """SVI Kalman delta ported from Kalman_SVI.cc and KalmanFilterJ_SVI.cc.

        State order is [X, VX, Y, VY]. The C++ wrapper uses diagonal Q
        (50, 10, 50, 10), confidence diagonal R, and its row-wise B*U
        operation for acceleration control.
        """
        st = self._kalman("SVI")
        z = np.array([position[0], speed[0], position[1], speed[1]], dtype=float)
        if not st["init"]:
            st["init"] = True
            st["x"] = z.copy()
            # Kalman_SVI::setInitial sets P0 to identity.
            st["P"] = np.eye(4)
            return np.zeros(4)

        T = float(time)
        A = np.array([
            [1.0, T, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, T],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        B = np.array([[0.5*T*T], [T], [0.5*T*T], [T]], dtype=float)
        # C++ MatrixOp_SVI::multiply1D computes M1[i][0] * M2[i].
        accel_xy = np.asarray(accel if accel is not None else (0.0, 0.0), dtype=float).reshape(2)
        U = np.array([accel_xy[0], accel_xy[0], accel_xy[1], accel_xy[1]], dtype=float)
        control = B[:, 0] * U
        Q = np.diag([50.0, 10.0, 50.0, 10.0])
        R = np.diag([
            float(pos_conf[0]), float(speed_conf[0]),
            float(pos_conf[1]), float(speed_conf[1]),
        ])

        x0, P0 = st["x"], st["P"]
        xp = A @ x0 + control
        Pp = A @ P0 @ A.T + Q
        try:
            K = Pp @ np.linalg.inv(Pp + R)
        except np.linalg.LinAlgError:
            K = Pp @ np.linalg.pinv(Pp + R)
        x = xp + K @ (z - xp)
        P = (np.eye(4) - K) @ Pp
        st["x"], st["P"] = x.copy(), P.copy()
        return np.abs(x - z)

    def KalmanPositionConsistancyCheck(self, curPosition, oldPosition, curPosConfidence, time, kalmanSI=None):
        initialized = kalmanSI.isInit() if kalmanSI is not None else self._kalman("SI_POS")["init"]
        if not initialized:
            if kalmanSI is None:
                st=self._kalman("SI_POS"); st["init"]=True; st["x"]=np.asarray(xy(curPosition),dtype=float); st["P"]=np.eye(2)*10.0
            return 1.0
        if time >= self.params.MAX_KALMAN_TIME:
            if kalmanSI is not None: kalmanSI.setInitial(*xy(curPosition))
            else:
                st=self._kalman("SI_POS"); st["init"]=True; st["x"]=np.asarray(xy(curPosition)); st["P"]=np.eye(2)*10
            return 1.0
        cx,cy=xy(curPosConfidence); cx=self._confidence(cx,self.params.KALMAN_MIN_POS_RANGE); cy=self._confidence(cy,self.params.KALMAN_MIN_POS_RANGE)
        delta = np.asarray(kalmanSI.getDeltaPos(time,*xy(curPosition),cx,cy),dtype=float) if kalmanSI is not None else self._si_delta("SI_POS",time,xy(curPosition),(cx,cy))
        score=1.0-float(np.linalg.norm(delta))/(4*self.params.MAX_KALMAN_POS_RANGE*cx*max(float(time),1e-12))
        return 0.0 if not np.isfinite(score) else max(0.0,score)

    def KalmanPositionAccConsistancyCheck(self, curPosition, curSpeed, curPosConfidence, time, kalmanSI=None):
        initialized = kalmanSI.isInit() if kalmanSI is not None else self._kalman("SI_POS_ACC")["init"]
        if not initialized:
            if kalmanSI is None:
                st=self._kalman("SI_POS_ACC"); st["init"]=True; st["x"]=np.asarray(xy(curPosition),dtype=float); st["P"]=np.eye(2)*10.0
            return 1.0
        if time >= self.params.MAX_KALMAN_TIME:
            if kalmanSI is not None: kalmanSI.setInitial(*xy(curPosition))
            else:
                st=self._kalman("SI_POS_ACC"); st["init"]=True; st["x"]=np.asarray(xy(curPosition)); st["P"]=np.eye(2)*10
            return 1.0
        cx,cy=xy(curPosConfidence); cx=self._confidence(cx,self.params.KALMAN_MIN_POS_RANGE); cy=self._confidence(cy,self.params.KALMAN_MIN_POS_RANGE)
        delta=np.asarray(kalmanSI.getDeltaPos(time,*xy(curPosition),*xy(curSpeed),cx,cy),dtype=float) if kalmanSI is not None else self._si_delta("SI_POS_ACC",time,xy(curPosition),(cx,cy),control=xy(curSpeed))
        score=1.0-float(np.linalg.norm(delta))/(4*self.params.MAX_KALMAN_POS_RANGE*cx*max(float(time),1e-12))
        return 0.0 if not np.isfinite(score) else max(0.0,score)

    def KalmanSpeedConsistancyCheck(self, curSpeed, curAccel, curSpeedConfidence, time, kalmanSI=None):
        initialized=kalmanSI.isInit() if kalmanSI is not None else self._kalman("SI_SPEED")["init"]
        if not initialized:
            if kalmanSI is None:
                st=self._kalman("SI_SPEED"); st["init"]=True; st["x"]=np.asarray(xy(curSpeed),dtype=float); st["P"]=np.eye(2)*10.0
            return 1.0
        if time >= self.params.MAX_KALMAN_TIME:
            if kalmanSI is not None: kalmanSI.setInitial(*xy(curSpeed))
            else:
                st=self._kalman("SI_SPEED"); st["init"]=True; st["x"]=np.asarray(xy(curSpeed)); st["P"]=np.eye(2)*10
            return 1.0
        cx,cy=xy(curSpeedConfidence); cx=self._confidence(cx,self.params.KALMAN_MIN_SPEED_RANGE); cy=self._confidence(cy,self.params.KALMAN_MIN_SPEED_RANGE)
        delta=np.asarray(kalmanSI.getDeltaPos(time,*xy(curSpeed),*xy(curAccel),cx,cy),dtype=float) if kalmanSI is not None else self._si_delta("SI_SPEED",time,xy(curSpeed),(cx,cy),control=xy(curAccel))
        score=1.0-float(np.linalg.norm(delta))/(self.params.MAX_KALMAN_SPEED_RANGE*cx*max(float(time),1e-12))
        return 0.0 if not np.isfinite(score) else max(0.0,score)

    def KalmanPositionSpeedConsistancyCheck(self, curPosition, curPositionConfidence, curSpeed, curAccel, curSpeedConfidence, time, kalmanSVI=None):
        initialized=kalmanSVI.isInit() if kalmanSVI is not None else self._kalman("SVI")["init"]
        if not initialized:
            if kalmanSVI is None:
                st=self._kalman("SVI"); st["init"]=True; st["x"]=np.array([xy(curPosition)[0],xy(curSpeed)[0],xy(curPosition)[1],xy(curSpeed)[1]],dtype=float); st["P"]=np.eye(4)*10.0
            return (1.0,1.0)
        if time >= self.params.MAX_KALMAN_TIME:
            if kalmanSVI is not None: kalmanSVI.setInitial(*xy(curPosition),*xy(curSpeed))
            else:
                st=self._kalman("SVI"); st["init"]=True; st["x"]=np.array([xy(curPosition)[0],xy(curSpeed)[0],xy(curPosition)[1],xy(curSpeed)[1]]); st["P"]=np.eye(4)*10
            return (1.0,1.0)
        px,py=xy(curPositionConfidence); vx,vy=xy(curSpeedConfidence)
        px=self._confidence(px,self.params.KALMAN_MIN_POS_RANGE); py=self._confidence(py,self.params.KALMAN_MIN_POS_RANGE); vx=self._confidence(vx,self.params.KALMAN_MIN_SPEED_RANGE); vy=self._confidence(vy,self.params.KALMAN_MIN_SPEED_RANGE)
        delta=np.asarray(kalmanSVI.getDeltaPos(time,*xy(curPosition),*xy(curSpeed),*xy(curAccel),px,py,vx,vy),dtype=float) if kalmanSVI is not None else self._svi_delta(time,xy(curPosition),xy(curSpeed),xy(curAccel),(px,py),(vx,vy))
        denom_p=self.params.MAX_KALMAN_POS_RANGE*px*max(float(time),1e-12); denom_v=self.params.MAX_KALMAN_SPEED_RANGE*vx*max(float(time),1e-12)
        scores=(1.0-float(np.hypot(delta[0],delta[2]))/denom_p,1.0-float(np.hypot(delta[1],delta[3]))/denom_v)
        return tuple(0.0 if not np.isfinite(v) else max(0.0,v) for v in scores)

    def KalmanPositionSpeedScalarConsistancyCheck(self, curPosition, oldPosition, curPositionConfidence, curSpeed, curAccel, curSpeedConfidence, time, kalmanSC=None):
        initialized=kalmanSC.isInit() if kalmanSC is not None else self._kalman("SC")["init"]
        if not initialized:
            if kalmanSC is None:
                curspd=float(np.linalg.norm(xy(curSpeed))); st=self._kalman("SC"); st["init"]=True; st["x"]=np.array([0.0,curspd]); st["P"]=np.eye(2)*10.0
            return (1.0,1.0)
        if time >= self.params.MAX_KALMAN_TIME:
            curspd=float(np.linalg.norm(xy(curSpeed)))
            if kalmanSC is not None: kalmanSC.setInitial(0,curspd)
            else:
                st=self._kalman("SC"); st["init"]=True; st["x"]=np.array([0.0,curspd]); st["P"]=np.eye(2)*10
            return (1.0,1.0)
        d=distance(curPosition,oldPosition); curspd=float(np.linalg.norm(xy(curSpeed))); curacl=float(np.linalg.norm(xy(curAccel)))
        px=self._confidence(xy(curPositionConfidence)[0],self.params.KALMAN_MIN_POS_RANGE); vx=self._confidence(xy(curSpeedConfidence)[0],self.params.KALMAN_MIN_SPEED_RANGE)
        delta=np.asarray(kalmanSC.getDeltaPos(time,d,curspd,curacl,curacl,px,vx),dtype=float) if kalmanSC is not None else self._sc_delta(time,d,curspd,curacl,px,vx)
        denom_p=self.params.MAX_KALMAN_POS_RANGE*px*max(float(time),1e-12); denom_v=self.params.MAX_KALMAN_SPEED_RANGE*vx*max(float(time),1e-12)
        scores=(1.0-delta[0]/denom_p,1.0-delta[1]/denom_v)
        return tuple(0.0 if not np.isfinite(v) else max(0.0,float(v)) for v in scores)


# ---- End integrated F2MD plausibility-check functions ----

FEATURES = [
    "sender_spd", "sender_acl", "time_delta", "distance", "heading_change",
    "speed_consistency", "accel_consistency", "heading_trajectory_consistency",
    "tx_delay", "mean_sender_receiver_dist", "n_receivers",
    "sender_dist_to_road_edge",
]
LABEL = "attacker"
ALIASES = {
    "scenario": ["scenario", "scenario_id", "scenarioid", "scene"],
    "sender": ["sender", "sender_id", "station_id", "vehicle_id", "pseudonym", "pseudonym_id"],
    "time": ["sendTime", "send_time", "timestamp", "time", "sim_time", "event_time", "message_time"],
    "x": ["x", "sender_pos_x", "pos_x", "position_x", "sender_x", "x_position"],
    "y": ["y", "sender_pos_y", "pos_y", "position_y", "sender_y", "y_position"],
    "receiver_x": ["receiver_x", "rx", "receiver_mean_pos_x", "receiver_pos_x", "receiver_position_x", "x_receiver", "x_rx"],
    "receiver_y": ["receiver_y", "ry", "receiver_mean_pos_y", "receiver_pos_y", "receiver_position_y", "y_receiver", "y_ry"],
    "speed": ["speed", "sender_spd", "velocity", "sender_speed"],
}


def read_table(path: Path) -> pd.DataFrame:
    """Read Excel (.xls/.xlsx) or CSV based on extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx", ".xlsm"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input extension {suffix!r}; use .xls, .xlsx, or .csv")


def detect_col(df: pd.DataFrame, aliases: list[str]):
    cols = {str(c).strip().lower(): c for c in df.columns}
    return next((cols[a.lower()] for a in aliases if a.lower() in cols), None)


def numeric_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    names = [f"{prefix}{feature}" for feature in FEATURES]
    missing = [name for name in names if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    out = df[names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out.columns = FEATURES
    return out


def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    delta = x - mean
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, inv_cov, delta), 0))


def bool_success(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "successful"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="randomforest_adversarial_examples.xls",
                        help="RandomForest adversarial examples file (.xls/.xlsx/.csv)")
    parser.add_argument("--data_dir", default="data/processed",
                        help="Directory containing train.parquet and test.parquet")
    parser.add_argument("--output_dir", default="physical_check_output")
    parser.add_argument("--quantile", type=float, default=0.99,
                        help="Quantile threshold for normal-class Mahalanobis distance")
    parser.add_argument("--bounds_tolerance", type=float, default=0.0,
                        help="Fraction of Q99-Q01 range allowed beyond feature bounds")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Absolute tolerance for formula and geometry residuals")
    parser.add_argument("--max_speed_mps", type=float, default=55.56,
                        help="Maximum derived trajectory speed, m/s (default ~200 km/h)")
    parser.add_argument("--reported_speed_tolerance_mps", type=float, default=8.0,
                        help="Allowed difference between reported and derived speed, m/s")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        # Helpful fallback: use a matching CSV/XLSX if requested .xls is absent.
        candidates = [input_path.with_suffix(ext) for ext in (".xlsx", ".csv", ".xls")]
        input_path = next((candidate for candidate in candidates if candidate.exists()), input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_examples = read_table(input_path)
    required_cols = {"attack_success", "source_row"}
    absent = required_cols - set(all_examples.columns)
    if absent:
        raise ValueError(f"Input is missing required columns: {sorted(absent)}")

    successful = all_examples[bool_success(all_examples["attack_success"])].copy().reset_index(drop=True)
    if successful.empty:
        raise ValueError("No rows with attack_success=True were found in the input file.")
    print(f"Read {len(all_examples)} total attack rows; extracted {len(successful)} successful attacks.")

    # Initialize each check as unavailable, and never treat unavailable checks as a pass.
    checks = [
        "source_row_valid", "feature_bounds_pass", "normal_mahalanobis_pass",
        "euclidean_distance_consistency_pass", "trajectory_dynamics_pass",
        "speed_consistency_formula_pass", "accel_consistency_formula_pass",
        "heading_trajectory_consistency_formula_pass",
        "heading_trajectory_dependency_heuristic_pass",
    ]
    for check in checks:
        successful[check] = pd.NA
    successful["check_availability_notes"] = ""

    data_dir = Path(args.data_dir)
    train_path = next((p for p in [data_dir / "train.csv", data_dir / "train.parquet"] if p.exists()), None)
    test_path = next((p for p in [data_dir / "test(1).csv", data_dir / "test.csv", data_dir / "test.parquet"] if p.exists()), None)
    train = test = None

    def load_dataset(path):
        if path is None:
            return None
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            warnings.warn(f"Parquet support unavailable ({exc}). Install pyarrow or fastparquet.")
            return None

    train = load_dataset(train_path)
    test = load_dataset(test_path)

    # Statistical checks: feature Q01-Q99 range and Normal-class Mahalanobis distance.
    if train is not None and LABEL in train.columns:
        train_x = numeric_features(train)
        medians = train_x.median()
        train_x = train_x.fillna(medians)
        normal_x = train_x.loc[pd.to_numeric(train[LABEL], errors="coerce").eq(0)]
        if len(normal_x) >= 2:
            q01, q99 = train_x.quantile(.01), train_x.quantile(.99)
            scale = (q99 - q01).replace(0, 1e-9)
            lower = q01 - args.bounds_tolerance * scale
            upper = q99 + args.bounds_tolerance * scale
            n = normal_x.to_numpy(float)
            mean = n.mean(axis=0)
            cov = np.cov(n, rowvar=False)
            ridge = max(float(np.trace(cov) / len(FEATURES)) * 1e-6, 1e-9)
            inv_cov = np.linalg.pinv(cov + np.eye(len(FEATURES)) * ridge)
            threshold = float(np.quantile(mahalanobis_distance(n, mean, inv_cov), args.quantile))

            adv_x = numeric_features(successful, prefix="adv_").fillna(medians)
            bounds_pass = ((adv_x >= lower) & (adv_x <= upper)).all(axis=1)
            md = mahalanobis_distance(adv_x.to_numpy(float), mean, inv_cov)
            md_pass = md <= threshold
            successful["feature_bounds_pass"] = bounds_pass
            successful["normal_mahalanobis_distance"] = md
            successful["normal_mahalanobis_threshold"] = threshold
            successful["normal_mahalanobis_pass"] = md_pass
        else:
            successful["check_availability_notes"] += "Insufficient Normal training rows for Mahalanobis; "
    else:
        successful["check_availability_notes"] += "train.parquet unavailable or lacks attacker label; statistical checks unavailable; "

    # Original source-row mapping enables geometric and sequence checks.
    if test is None:
        successful["check_availability_notes"] += "test.parquet unavailable; coordinate/trajectory/formula checks unavailable; "
    else:
        source = pd.to_numeric(successful["source_row"], errors="coerce")
        # source_row is a data column created by preprocessing; it is not necessarily
        # the dataframe row index after CSV export/shuffling. Map by that key when present.
        if "source_row" in test.columns:
            source_key = pd.to_numeric(test["source_row"], errors="coerce")
            source_lookup = test.assign(_source_row_numeric=source_key).dropna(subset=["_source_row_numeric"])
            source_lookup = source_lookup.drop_duplicates("_source_row_numeric").set_index("_source_row_numeric", drop=False)
            valid = source.notna() & source.isin(source_lookup.index)
            raw = source_lookup.reindex(source.where(valid).to_numpy()).reset_index(drop=True)
        else:
            valid = source.notna() & source.astype("Int64").isin(test.index)
            raw = test.reindex(source.where(valid).astype("Int64").to_numpy()).reset_index(drop=True)
        successful["source_row_valid"] = valid.to_numpy()
        detected = {key: detect_col(test, aliases) for key, aliases in ALIASES.items()}

        # Receiver-distance consistency: compare the adversarial feature to the
        # same preprocessing feature in the source row. This avoids the invalid
        # comparison between trajectory displacement and sender-to-mean-receiver
        # coordinate distance (and avoids recomputing a mean of distances from a
        # distance-to-mean-coordinate approximation).
        receiver_distance_feature = detect_col(test, ["mean_sender_receiver_dist"])
        if receiver_distance_feature and "adv_mean_sender_receiver_dist" in successful.columns:
            raw_receiver_dist = pd.to_numeric(raw[receiver_distance_feature], errors="coerce")
            adv_receiver_dist = pd.to_numeric(successful["adv_mean_sender_receiver_dist"], errors="coerce")
            residual = (adv_receiver_dist - raw_receiver_dist).abs()
            successful["recomputed_euclidean_distance"] = raw_receiver_dist
            successful["euclidean_distance_residual"] = residual
            successful["euclidean_distance_consistency_pass"] = residual.le(args.tolerance) & np.isfinite(residual)
        else:
            successful["check_availability_notes"] += "Source mean_sender_receiver_dist unavailable for like-for-like Euclidean check; "

        # Exact sequential engineered-feature formula checks and dependency heuristic.
        seq_available = all(detected[key] for key in ["scenario", "sender", "x", "y", "speed"])
        heading_col = detect_col(test, ["sender_hed", "sender_heading", "heading", "hed"])
        accel_col = detect_col(test, ["sender_acl", "sender_accel", "acceleration", "acl"])
        sendtime_col = detect_col(test, ["sendTime", "send_time"])
        seq_available = seq_available and bool(heading_col and accel_col and sendtime_col)
        if seq_available:
            base = test.copy()
            group_cols = [detected["scenario"], detected["sender"]]
            numeric_cols = [sendtime_col, detected["x"], detected["y"], detected["speed"], accel_col, heading_col]
            for col in numeric_cols:
                base[col] = pd.to_numeric(base[col], errors="coerce")
            base["_source_index"] = pd.to_numeric(base["source_row"], errors="coerce") if "source_row" in base.columns else base.index
            base = base.sort_values(group_cols + [sendtime_col], kind="stable")
            grouped = base.groupby(group_cols, dropna=False, sort=False)
            for field, col in [("_prev_speed", detected["speed"]), ("_prev_x", detected["x"]),
                               ("_prev_y", detected["y"]), ("_prev_time", sendtime_col)]:
                base[field] = grouped[col].shift(1)
            state = base.set_index("_source_index").reindex(source.where(valid).astype("Int64").to_numpy()).reset_index(drop=True)
            # Formula validation uses the adversarial values for current-row
            # engineered inputs, and the true preceding source-row speed for
            # acceleration. The preprocessing threshold is 0.05 seconds.
            adv_speed = pd.to_numeric(successful["adv_sender_spd"], errors="coerce")
            adv_accel = pd.to_numeric(successful["adv_sender_acl"], errors="coerce")
            adv_dist = pd.to_numeric(successful["adv_distance"], errors="coerce")
            adv_dt = pd.to_numeric(successful["adv_time_delta"], errors="coerce")
            reliable = adv_dt.ge(0.05) & np.isfinite(adv_dt)

            expected_speed_consistency = np.where(reliable, adv_speed - adv_dist / adv_dt.where(reliable), 0.0)
            actual_speed_consistency = pd.to_numeric(successful["adv_speed_consistency"], errors="coerce")
            speed_residual = (actual_speed_consistency - expected_speed_consistency).abs()
            successful["speed_consistency_formula_residual"] = speed_residual
            successful["speed_consistency_formula_pass"] = speed_residual.le(args.tolerance) & np.isfinite(speed_residual)

            prev_speed = pd.to_numeric(state["_prev_speed"], errors="coerce")
            has_prev = prev_speed.notna()
            derived_accel = np.where(reliable & has_prev, (adv_speed - prev_speed) / adv_dt.where(reliable & has_prev), 0.0)
            expected_accel_consistency = np.where(reliable & has_prev, adv_accel - derived_accel, 0.0)
            actual_accel_consistency = pd.to_numeric(successful["adv_accel_consistency"], errors="coerce")
            accel_residual = (actual_accel_consistency - expected_accel_consistency).abs()
            successful["accel_consistency_formula_residual"] = accel_residual
            successful["accel_consistency_formula_pass"] = accel_residual.le(args.tolerance) & np.isfinite(accel_residual)

            dx = pd.to_numeric(raw[detected["x"]], errors="coerce") - pd.to_numeric(state["_prev_x"], errors="coerce")
            dy = pd.to_numeric(raw[detected["y"]], errors="coerce") - pd.to_numeric(state["_prev_y"], errors="coerce")
            displacement = np.sqrt(dx ** 2 + dy ** 2)
            movement_heading = (90.0 - np.degrees(np.arctan2(dy, dx))) % 360.0
            reported_heading = pd.to_numeric(raw[heading_col], errors="coerce") % 360.0
            angle_diff = (reported_heading - movement_heading).abs()
            expected_heading_consistency = np.minimum(angle_diff, 360.0 - angle_diff)
            expected_heading_consistency = expected_heading_consistency.where(displacement >= 0.1, 0.0)
            expected_heading_consistency = expected_heading_consistency.where(state["_prev_x"].notna() & state["_prev_y"].notna(), 0.0)
            actual_heading_consistency = pd.to_numeric(successful["adv_heading_trajectory_consistency"], errors="coerce")
            heading_residual = (actual_heading_consistency - expected_heading_consistency).abs()
            successful["heading_trajectory_consistency_formula_residual"] = heading_residual
            successful["heading_trajectory_consistency_formula_pass"] = heading_residual.le(args.tolerance) & np.isfinite(heading_residual)
            successful["heading_trajectory_consistency_formula_inconsistency_flag"] = ~successful["heading_trajectory_consistency_formula_pass"].astype(bool)
            successful["sequential_formula_checks_available"] = True
        else:
            successful["check_availability_notes"] += "Required raw columns for exact sequential formula checks unavailable; "
            successful["sequential_formula_checks_available"] = False

        # Dependency heuristic flags changed heading trajectory consistency if proxy fields
        # heading_change and time_delta are both unmodified. Formula recomputation is stronger.
        htc_changed = pd.to_numeric(successful.get("delta_heading_trajectory_consistency"), errors="coerce").abs().gt(args.tolerance)
        proxy_inputs_changed = pd.concat([
            pd.to_numeric(successful.get("delta_heading_change"), errors="coerce").abs().gt(args.tolerance),
            pd.to_numeric(successful.get("delta_time_delta"), errors="coerce").abs().gt(args.tolerance),
        ], axis=1).any(axis=1)
        dependency_flag = htc_changed & ~proxy_inputs_changed
        successful["heading_trajectory_dependency_inconsistency_flag"] = dependency_flag
        successful["heading_trajectory_dependency_heuristic_pass"] = ~dependency_flag
        if "heading_trajectory_consistency_formula_inconsistency_flag" in successful.columns:
            successful["heading_trajectory_consistency_inconsistency_flag"] = (
                successful["heading_trajectory_consistency_formula_inconsistency_flag"] | dependency_flag
            )

        # Physical trajectory dynamics is calculated over the FULL source test
        # table, not merely the subset of successful adversarial source rows.
        # Results are then mapped back using source_row to preserve predecessor
        # context and avoid false jumps between sparsely selected attack rows.
        trajectory_available = all(detected[key] for key in ["sender", "time", "x", "y"])
        if trajectory_available and "source_row" in test.columns:
            full = test.copy()
            group = [detected["sender"]]
            if detected["scenario"]:
                group.insert(0, detected["scenario"])
            time_col, x_col, y_col = detected["time"], detected["x"], detected["y"]
            numeric = [time_col, x_col, y_col] + ([detected["speed"]] if detected["speed"] else [])
            for col in numeric:
                full[col] = pd.to_numeric(full[col], errors="coerce")
            full["_source_row_key"] = pd.to_numeric(full["source_row"], errors="coerce")
            full = full.sort_values(group + [time_col], kind="stable")
            fg = full.groupby(group, dropna=False, sort=False)
            dt_dyn = fg[time_col].diff() * 1e-9
            dx_dyn, dy_dyn = fg[x_col].diff(), fg[y_col].diff()
            derived_speed = np.sqrt(dx_dyn ** 2 + dy_dyn ** 2) / dt_dyn.where(dt_dyn > 0)
            dynamics_ok = dt_dyn.gt(0) & np.isfinite(derived_speed) & derived_speed.le(args.max_speed_mps)
            if detected["speed"]:
                reported = full[detected["speed"]]
                dynamics_ok &= reported.isna() | ((reported - derived_speed).abs() <= args.reported_speed_tolerance_mps)
            full["_dynamics_ok"] = dynamics_ok
            dynamics_lookup = full.dropna(subset=["_source_row_key"]).drop_duplicates("_source_row_key").set_index("_source_row_key")["_dynamics_ok"]
            selected_keys = pd.to_numeric(successful["source_row"], errors="coerce")
            successful["trajectory_dynamics_pass"] = selected_keys.map(dynamics_lookup).astype("boolean")
            successful.loc[successful["trajectory_dynamics_pass"].isna(), "check_availability_notes"] += "No predecessor/source trajectory record for dynamics; "
        else:
            successful["check_availability_notes"] += "Sender/time/x/y/source_row columns unavailable for trajectory dynamics check; "

    # Report a graded assessment rather than a brittle all-checks-must-pass
    # binary verdict. Statistical screens and the heading dependency heuristic
    # are advisory; the primary score reflects the available formula/dynamics
    # checks. Missing checks remain explicitly indeterminate.
    all_check_cols = [
        "feature_bounds_pass", "normal_mahalanobis_pass", "euclidean_distance_consistency_pass",
        "trajectory_dynamics_pass", "speed_consistency_formula_pass", "accel_consistency_formula_pass",
        "heading_trajectory_consistency_formula_pass", "heading_trajectory_dependency_heuristic_pass",
    ]
    primary_check_cols = [
        "euclidean_distance_consistency_pass", "trajectory_dynamics_pass",
        "speed_consistency_formula_pass", "accel_consistency_formula_pass",
        "heading_trajectory_consistency_formula_pass",
    ]
    successful["checks_available_count"] = successful[all_check_cols].notna().sum(axis=1)
    successful["checks_failed_count"] = successful[all_check_cols].apply(
        lambda row: sum(value is False or (isinstance(value, (np.bool_, bool)) and not value) for value in row), axis=1
    )
    successful["all_available_checks_pass"] = successful[all_check_cols].apply(
        lambda row: all(pd.isna(value) or bool(value) for value in row), axis=1
    )
    successful["complete_check_coverage"] = successful[all_check_cols].notna().all(axis=1)
    successful["primary_checks_available_count"] = successful[primary_check_cols].notna().sum(axis=1)
    successful["primary_checks_pass_count"] = successful[primary_check_cols].apply(
        lambda row: sum(bool(value) for value in row if pd.notna(value)), axis=1
    )
    successful["primary_checks_pass_rate"] = (
        successful["primary_checks_pass_count"] / successful["primary_checks_available_count"].replace(0, np.nan)
    )
    successful["primary_checks_complete_coverage"] = successful[primary_check_cols].notna().all(axis=1)
    successful["physical_plausibility_assessment"] = np.select(
        [
            successful["primary_checks_available_count"].eq(0),
            ~successful["primary_checks_complete_coverage"],
            successful["primary_checks_pass_rate"].ge(0.8),
            successful["primary_checks_pass_rate"].lt(0.5),
        ],
        ["Indeterminate - no primary checks available", "Indeterminate - partial primary coverage",
         "Mostly consistent - full primary coverage", "Potential inconsistencies - primary checks"],
        default="Mixed evidence - review failed checks",
    )
    # Retain legacy boolean as a conservative indicator, but do not treat it as
    # the headline scientific conclusion. It requires every check to be present.
    successful["physically_plausible_screen"] = (
        successful["primary_checks_complete_coverage"] & successful["primary_checks_pass_rate"].eq(1.0)
    )

    # Export all extracted successes, with full columns and additional check results.
    csv_out = output_dir / "randomforest_successful_attacks_physical_checks.csv"
    xlsx_out = output_dir / "randomforest_successful_attacks_physical_checks.xlsx"
    summary_out = output_dir / "physical_plausibility_summary.json"
    successful.to_csv(csv_out, index=False)
    summary = {
        "input_file": str(input_path),
        "input_rows": int(len(all_examples)),
        "successful_attack_rows_extracted": int(len(successful)),
        "successful_rows_passing_all_available_checks": int(successful["all_available_checks_pass"].sum()),
        "successful_rows_passing_all_checks_with_complete_coverage": int(successful["physically_plausible_screen"].sum()),
        "physical_plausibility_assessment_counts": successful["physical_plausibility_assessment"].value_counts(dropna=False).to_dict(),
        "mean_primary_checks_pass_rate": float(successful["primary_checks_pass_rate"].mean()),
        "check_columns": all_check_cols,
        "check_coverage_complete_count": int(successful["complete_check_coverage"].sum()),
        "notes": "A check marked NA is unavailable. Headline assessment is graded on primary geometry/trajectory/formula checks; statistical bounds/Mahalanobis and dependency heuristic are advisory. Legacy physically_plausible_screen is a conservative boolean, not the sole scientific conclusion.",
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        successful.to_excel(writer, sheet_name="Successful Attacks Checks", index=False)
    print(f"Extracted successful attacks: {len(successful)} / {len(all_examples)}")
    print(f"CSV results: {csv_out}")
    print(f"Excel results: {xlsx_out}")
    print(f"Summary JSON: {summary_out}")


if __name__ == "__main__":
    main()
