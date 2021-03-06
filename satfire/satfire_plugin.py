#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017, Panu Lahtinen / FMI
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>

"""Satfire plugin for Trollflow"""

import logging
import time

from trollflow.workflow_component import AbstractWorkflowComponent
from trollflow.utils import acquire_lock, release_lock
from satfire.forest_fire import ForestFire


class Satfire(AbstractWorkflowComponent):

    """Do something"""

    logger = logging.getLogger("Satfire")

    def __init__(self):
        super(Satfire, self).__init__()

    def pre_invoke(self):
        """Pre-invoke"""
        pass

    def invoke(self, context):
        """Invoke"""
        # Set locking status, default to False
        self.use_lock = context.get("use_lock", False)
        self.logger.debug("Locking is used in Satfire: %s",
                          str(self.use_lock))
        if self.use_lock:
            self.logger.debug("Satfire acquires lock of previous "
                              "worker: %s", str(context["prev_lock"]))
            acquire_lock(context["prev_lock"])

        self.logger.info("Finding forest fires.")

        fff = ForestFire(context["config"])
        try:
            if fff.run(msg=context["content"]):
                if "text_fname_pattern" in context["config"]:
                    fff.save_text()
                if "hdf5_fname_pattern" in context["config"]:
                    fff.save_hdf5()
            fff.clean()
        finally:
            if fff._pub is not None:
                fff._pub.stop()

        if self.use_lock:
            self.logger.debug("Satfire releases own lock %s",
                              str(context["lock"]))
            release_lock(context["lock"])
            # Wait 1 second to ensure next worker has time to acquire the
            # lock
            time.sleep(1)

        # Wait until the lock has been released downstream
        if self.use_lock:
            acquire_lock(context["lock"])
            release_lock(context["lock"])

        # After all the items have been processed, release the lock for
        # the previous step
        self.logger.debug("Scene loader releses lock of previous worker")
        release_lock(context["prev_lock"])

    def post_invoke(self):
        """Post-invoke"""
        pass
