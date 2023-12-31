/*
 * Copyright (C) from 2022 The Play Framework Contributors <https://github.com/playframework>, 2011-2021 Lightbend Inc. <https://www.lightbend.com>
 */

package play.api.test

import org.specs2.mutable.SpecificationLike
import play.api.http.HeaderNames
import play.api.http.HttpProtocol
import play.api.http.HttpVerbs
import play.api.http.Status

/**
 * Play specs2 specification.
 *
 * This trait excludes some of the mixins provided in the default specs2 specification that clash with Play helpers
 * methods.  It also mixes in the Play test helpers and types for convenience.
 */
trait PlaySpecification
    extends SpecificationLike
    with PlayRunners
    with HeaderNames
    with Status
    with HttpProtocol
    with DefaultAwaitTimeout
    with ResultExtractors
    with Writeables
    with RouteInvokers
    with FutureAwaits
    with HttpVerbs {}
