/*
 * Copyright (C) from 2022 The Play Framework Contributors <https://github.com/playframework>, 2011-2021 Lightbend Inc. <https://www.lightbend.com>
 */

package views.html.helper

import play.api.mvc._
import play.twirl.api.Html
import play.twirl.api.HtmlFormat

/**
 * CSRF helper for Play calls
 */
object CSRF {
  def getToken(implicit request: RequestHeader): play.filters.csrf.CSRF.Token =
    play.filters.csrf.CSRF.getToken.getOrElse(
      sys.error("No CSRF token was generated for this request! Is the CSRF filter installed?")
    )

  /**
   * Add the CSRF token as a query String parameter to this reverse router request
   */
  def apply(call: Call)(implicit request: RequestHeader): Call = {
    val token = getToken
    new Call(
      call.method,
      s"${call.url}${if (call.url.contains("?")) "&" else "?"}${token.name}=${token.value}"
    )
  }

  /**
   * Render a CSRF form field token
   */
  def formField(implicit request: RequestHeader): Html = {
    val token = getToken
    // probably not possible for an attacker to XSS with a CSRF token, but just to be on the safe side...
    Html(s"""<input type="hidden" name="${token.name}" value="${HtmlFormat.escape(token.value)}"/>""")
  }
}
